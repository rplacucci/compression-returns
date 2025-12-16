import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import argparse
import torch
import evaluate
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, logging, get_scheduler
from .utils import tokenize_fn, postprocess_fn

# accelerate launch -m src.tune 

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a task")
parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="Name of of pre-trained BERT model")
parser.add_argument("--task_name", type=str, default="ag_news", choices=["ag_news", "dbpedia_14"], help="Name of task to tune on")
parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for the optimizer")
parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999), help="Beta values for the optimizer")
parser.add_argument("--eps", type=float, default=1e-6, help="Constant to stabilize division in the optimizer update rule")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Size of batch to train with")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Initial fraction of training steps with linear LR warmup")
parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor for cross-entropy loss")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments")
args = parser.parse_args()

checkpoint = args.checkpoint
task_name = args.task_name
lr = args.lr
betas = args.betas
eps = args.eps
weight_decay = args.weight_decay
batch_size = args.batch_size
grad_accum_steps = args.grad_accum_steps
warmup_ratio = args.warmup_ratio
label_smoothing = args.label_smoothing
n_epochs = args.n_epochs
seed = args.seed

# Config directories
run_id = f"{checkpoint}-{task_name}-lr-{lr:.0e}-batch_size-{batch_size:02d}-grad_accum_steps-{grad_accum_steps:02d}-warmup_ratio-{warmup_ratio:.2f}-label_smoothing-{label_smoothing:.2f}-n_epochs-{n_epochs}-seed-{seed:02d}"

log_dir = f"./logs/{task_name}"
os.makedirs(log_dir, exist_ok=True)

out_dir = f"./models/{task_name}"
os.makedirs(out_dir, exist_ok=True)
model_save_path = os.path.join(out_dir, run_id)

# Explicitly set CUDA device
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)

# Config distributed training with accelerate
accelerator = Accelerator(
    log_with=["tensorboard"],
    project_dir=log_dir,
    device_placement=True,
    gradient_accumulation_steps=grad_accum_steps,
    mixed_precision="bf16"
)
accelerator.init_trackers(run_id)
world_size = accelerator.num_processes
accelerator.print(f"Initialized {accelerator.__class__.__name__} with {world_size} distributed processes and {grad_accum_steps} gradient accumulation steps")

# Set seeds for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Config tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
vocab_size = tokenizer.vocab_size
accelerator.print(f"Loaded {tokenizer.__class__.__name__} with vocab size {vocab_size:,}")

# Config model
if task_name == "ag_news":
    num_labels = 4
elif task_name == "dbpedia_14":
    num_labels = 14
else:
    raise NotImplementedError(f"Task '{task_name}' is not supported")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
accelerator.print(f"Loaded {model.__class__.__name__} with {num_labels} labels and {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Load dataset
ds = load_dataset(f"fancyzhx/{task_name}")
accelerator.print(f"Loaded {task_name} with {len(ds['train']):,} train and {len(ds['test']):,} test examples")

dataset = ds.map(tokenize_fn, batched=True, fn_kwargs={"task_name": task_name, "tokenizer": tokenizer})
dataset = postprocess_fn(dataset, task_name)

# Prepare dataloaders
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    dataset=dataset['train'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

valid_dataloader = DataLoader(
    dataset=dataset['test'],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# Load evaluation metric
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1", average="macro")

# Config optimizer and distributed learning
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Prepare distributed learning
train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, valid_dataloader, model, optimizer
)

# Config LR scheduler
total_steps = n_epochs * len(train_dataloader) // grad_accum_steps
warmup_steps = int(warmup_ratio * total_steps)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=total_steps,
    num_warmup_steps=warmup_steps
)

# Config loss function
criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

# Begin training
accelerator.print(f"Begin training with an effective batch size of {batch_size * grad_accum_steps}...")
log_steps = 10
step = 0
for epoch in range(n_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        start = time.time()

        with accelerator.accumulate(model):
            outputs = model(**batch)
            
            if label_smoothing > 0.0:
                loss = criterion(outputs.logits, batch['labels'])
            else:
                loss = outputs.loss

            accelerator.backward(loss)
            
            # Only step optimizer when accumulation is complete
            if accelerator.sync_gradients:
                norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = time.time() - start
        max_len = batch["input_ids"].shape[1]
        tokens_per_sec = int(batch_size * max_len * world_size / elapsed) if elapsed > 0 else 0

        # Only log when gradients are synchronized (i.e., after accumulation steps)
        if accelerator.sync_gradients:
            loss_item = loss.detach().item()
            lr = scheduler.get_last_lr()[0]
            
            # Log to tensorboard
            if step % log_steps == 0:
                accelerator.log(
                    {
                        "loss": loss_item,
                        "grad_norm": norm,
                        "lr": lr,
                    },
                    step=step,
                )

            # Print to terminal
            accelerator.print(f"(train) epoch: {epoch:2d} | step: {step:4d} | loss: {loss_item:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tokens_per_sec:,}")

    model.eval()
    for batch in valid_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Gather preds and refs across processes then add to metric
        preds = accelerator.gather_for_metrics(predictions)
        refs = accelerator.gather_for_metrics(batch["labels"])
        acc_metric.add_batch(predictions=preds, references=refs)
        f1_metric.add_batch(predictions=preds, references=refs)
    
    # Log to tensorboard
    acc_score = acc_metric.compute()
    f1_score = f1_metric.compute(average="macro")
    
    scores = {**acc_score, **f1_score}
    accelerator.log(scores, step=epoch)
    
    # Print to terminal
    scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
    accelerator.print(f"(valid) epoch: {epoch:2d} | {scores_str}")

# Save final model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(model_save_path, save_function=accelerator.save)
accelerator.print(f"Model saved to {model_save_path}")

# Flush trackers
accelerator.end_training()
accelerator.print("Goodbye!")
import os
import csv
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import evaluate
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, logging, get_scheduler
from .utils import tokenize_fn, postprocess_fn

# accelerate launch -m src.prune

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Config argparser
parser = argparse.ArgumentParser(description="Prune and evaluate fine-tuned BERT on task")
parser.add_argument("--checkpoint", type=str, required=True, choices=["bert-base-uncased", "bert-large-uncased"], help="Name of BERT checkpoint")
parser.add_argument("--task_name", type=str, required=True, choices=["ag_news", "dbpedia_14"], help="Name of downstream task")
parser.add_argument("--sparsity", type=float, required=True, help="Global fraction of weights to prune")
parser.add_argument("--recovery_epochs", type=int, default=0, help="Optional short recovery fine-tune after pruning")
parser.add_argument("--batch_size", type=int, default=32, help="Size of batches for train/eval")
parser.add_argument("--seed", type=int, default=42, help="Random seed for experiments")
args = parser.parse_args()

checkpoint = args.checkpoint
task_name = args.task_name
sparsity = args.sparsity
recovery_epochs = args.recovery_epochs
batch_size = args.batch_size
seed = args.seed

# Config directories
log_dir = f"./logs/pruning/{task_name}"
os.makedirs(log_dir, exist_ok=True)
run_id = f"{checkpoint}-{task_name}-sparsity-{sparsity:.1f}-recovery_epochs-{recovery_epochs:02d}-seed-{seed:02d}"

# Config distributed training with accelerate
if recovery_epochs > 0:
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 1e-2
    label_smoothing = 0.05 if task_name == "ag_news" else 0.10
    
    if checkpoint == "bert-base-uncased":
        grad_accum_steps = 4
        lr = 3e-5 / 2
        warmup_ratio = 0.06

    elif checkpoint == "bert-large-uncased":
        grad_accum_steps = 8
        lr = 2e-5 / 2
        warmup_ratio = 0.10

    else:
        raise NotImplementedError(f"Checkpoint '{checkpoint}' is not supported")

accelerator = Accelerator(
    log_with=["tensorboard"],
    project_dir=log_dir,
    device_placement=True,
    gradient_accumulation_steps=grad_accum_steps if recovery_epochs > 0 else 1
)
accelerator.init_trackers(run_id)
world_size = accelerator.num_processes
accelerator.print(f"Initialized {accelerator.__class__.__name__} with {world_size} distributed processes")

# Set seeds for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.set_float32_matmul_precision("high")

# Config tokenizer
tokenizer = BertTokenizer.from_pretrained(checkpoint)
vocab_size = tokenizer.vocab_size
accelerator.print(f"Loaded {tokenizer.__class__.__name__} from checkpoint {checkpoint} with vocab size {vocab_size:,}")

# Config model
if task_name == "ag_news":
    num_labels = 4
elif task_name == "dbpedia_14":
    num_labels = 14
else:
    raise NotImplementedError(f"Task '{task_name}' is not supported")

model_dir = f"./models/{task_name}/{checkpoint}"
model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
total_params = sum(p.numel() for p in model.parameters())
accelerator.print(f"Loaded {model.__class__.__name__} from {model_dir} with {num_labels} labels and {total_params:,} parameters")

# Prune weights from linear layers
accelerator.print(f"Pruning {sparsity:.0%} of linear layer parameters...")
to_prune = []
for module in model.modules():
    if isinstance(module, nn.Linear):
        to_prune.append((module, "weight"))

prune.global_unstructured(
    parameters=to_prune,
    pruning_method=prune.L1Unstructured,
    amount=sparsity
)

for module, name in to_prune:
    prune.remove(module, name)

linear_params_total = sum(p.numel() for module in model.modules() if isinstance(module, nn.Linear) for p in module.parameters())
linear_params_effective = sum((p != 0).sum().item() for module in model.modules() if isinstance(module, nn.Linear) for p in module.parameters())
sparsity_actual = 1 - (linear_params_effective / linear_params_total)
accelerator.print(f"Pruned {sparsity_actual:.3%} of linear layer parameters")

# Load dataset
ds = load_dataset(f"fancyzhx/{task_name}")
accelerator.print(f"Loaded {task_name} with {len(ds['train']):,} train and {len(ds['test']):,} test examples")

dataset = ds.map(tokenize_fn, batched=True, fn_kwargs={"task_name": task_name, "tokenizer": tokenizer})
dataset = postprocess_fn(dataset, task_name)

# Prepare dataloaders
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

if recovery_epochs > 0:
    train_dataloader = DataLoader(
        dataset=dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

valid_dataloader = DataLoader(
    dataset=dataset['test'],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Define evaluation logic
def evaluate_model(model, dataloader, accelerator):
    accelerator.print("Evaluating...")
    model.eval()
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1", average="macro")
    
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Gather preds and refs across processes then add to metric
        preds = accelerator.gather_for_metrics(predictions)
        refs = accelerator.gather_for_metrics(batch["labels"])
        acc_metric.add_batch(predictions=preds, references=refs)
        f1_metric.add_batch(predictions=preds, references=refs)

    # Compute and return scores
    acc_score = acc_metric.compute()
    f1_score = f1_metric.compute(average="macro")
    return {**acc_score, **f1_score}

# Config optimizer, LR scheduler, loss function, and distributed learning
if recovery_epochs > 0:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    total_steps = recovery_epochs * len(train_dataloader) // grad_accum_steps
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps
    )
    train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, valid_dataloader, model, optimizer
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
else:
    valid_dataloader, model = accelerator.prepare(valid_dataloader, model)

# Evaluate pruned model
scores = evaluate_model(model, valid_dataloader, accelerator)

# Print to terminal
scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
accelerator.print(f"(w/o recovery) {scores_str}")

# Recovery fine-tune
if recovery_epochs > 0:
    accelerator.print(f"Recovering...")
    for epoch in range(recovery_epochs):
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
                
                # Print to terminal
                accelerator.print(f"(train) epoch: {epoch:2d} | step: {step:4d} | loss: {loss_item:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tokens_per_sec:,}")

    # Evaluate pruned model after recovery
    scores_with_recovery = evaluate_model(model, valid_dataloader, accelerator)
    scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores_with_recovery.items()])
    accelerator.print(f"(w/ recovery) {scores_str}")

# Log experimental results
if accelerator.is_main_process:
    results = {
        "method": "pruning",
        "checkpoint": checkpoint,
        "task_name": task_name,
        "seed": seed,
        "sparsity_target": sparsity,
        "sparsity_actual": sparsity_actual,
        "recovery_epochs": recovery_epochs,
        "accuracy": scores['accuracy'],
        "f1": scores['f1'],
        "acc_with_recovery": scores_with_recovery['accuracy'] if recovery_epochs > 0 else None,
        "f1_with_recovery": scores_with_recovery['f1'] if recovery_epochs > 0 else None,
    }

    results_file = f"./outputs/experiments_pruning.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    file_exists = os.path.exists(results_file)
    
    with open(results_file, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    
    accelerator.print(f"Results saved to {results_file}")

# Flush trackers
accelerator.end_training()
accelerator.print("Goodbye!")
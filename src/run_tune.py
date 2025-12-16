import argparse
import subprocess
import math
import yaml
from itertools import product

# Define helper functions
def default_per_device_bs(hidden_size: int) -> int:
    # Heuristics that fit typical memory for BERT-miniatures
    if hidden_size <= 256:
        return 128
    if hidden_size <= 512:
        return 64
    return 32

def derive_grad_accum(effective_bs: int, per_device_bs: int, num_devices: int=1) -> int:
    denom = per_device_bs * max(1, num_devices)
    return max(1, math.ceil(effective_bs / denom))

# Config argparser
parser = argparse.ArgumentParser(description="Launch BERT fine-tuning sweep with accelerate")
parser.add_argument("--task_name", type=str, choices=["ag_news", "dbpedia_14"], help="Name of downstream task")
parser.add_argument("--sweep_yaml", type=str, default="./configs/sweeps.yaml", help="Path to sweep config YAML")
args = parser.parse_args()

task_name = args.task_name
sweep_yaml = args.sweep_yaml

# Load yaml config
with open(sweep_yaml, "r") as f:
    cfg = yaml.safe_load(f)

by_task = cfg[task_name]

# Define model attributes
num_hidden_layers = [
    2,
    4,
    6,
    8,
    10,
    12
]

hidden_size_attn_heads = [
    (128, 2),
    (256, 4),
    (512, 8),
    (768, 12),
]

launched = 0
for L in num_hidden_layers:
    for H, A in hidden_size_attn_heads:
        checkpoint = f"google/bert_uncased_L-{L}_H-{H}_A-{A}"

        by_hidden = by_task['by_hidden_size'].get(f"H{H}")
        if by_hidden is None:
            raise ValueError(f"No sweep for defined hidden size H={H}")

        pbs = default_per_device_bs(hidden_size=H)
        combos = list(product(
            by_hidden['learning_rate'],
            by_hidden['warmup_ratio'],
            by_task['label_smoothing_factor'],
            by_task['effective_batch_size'],
            by_task['num_train_epochs'],
        ))
        
        for (lr, wu, ls, ebs, epochs) in combos:
            cmd = [
                "accelerate", "launch", "-m", "src.tune",
                "--checkpoint", checkpoint,
                "--task_name", task_name,
                "--lr", str(lr),
                "--batch_size", str(pbs),
                "--grad_accum_steps", str(derive_grad_accum(ebs, pbs)),
                "--warmup_ratio", str(wu),
                "--label_smoothing", str(ls),
                "--n_epochs", str(epochs)
            ]

            print(">>>", " ".join(cmd))
            subprocess.run(cmd, check=True)
            launched += 1

print(f"[run_tune] Completed {launched} configuration(s).")
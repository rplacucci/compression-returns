import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# python -m src.plot_experiments

# Config argparser
parser = argparse.ArgumentParser(description="Plot quality and return graphs from experimental data")
parser.add_argument("--metric", type=str, required=True, choices=["accuracy", "f1"], help="Performance metric to evaluate model")
args = parser.parse_args()

metric = args.metric

# Read the CSV file into a DataFrame
experiments = pd.read_csv("./outputs/experiments_pruning.csv")

# Calculate normalization values per task from the data
normalization_values = {}
for task in experiments['task_name'].unique():
    task_data = experiments[experiments['task_name'] == task]
    min_val = task_data[metric].min()
    max_val = task_data[metric].max()
    normalization_values[task] = (min_val, max_val)

# Prepare plots
plt.figure(figsize=(10, 6))
quality_ax = plt.subplot(1, 2, 1)
returns_ax = plt.subplot(1, 2, 2)

auc_data = []

for checkpoint in experiments['checkpoint'].unique():
    for task_name in experiments['task_name'].unique():
        # Filter data for this combination
        subset = experiments[
            (experiments['checkpoint'] == checkpoint) & 
            (experiments['task_name'] == task_name)
        ]
        
        if subset.empty:
            continue
            
        sparsity = subset['sparsity_actual'].sort_values()
        
        # Get normalization values for this task
        min_val, max_val = normalization_values[task_name]
        
        quality = (subset[metric] - min_val) / (max_val - min_val)
        quality = np.clip(quality, 0, 1)  # Ensure values stay in [0,1]
        
        # Sort quality by sparsity for proper AUC calculation
        quality_sorted = quality.reindex(sparsity.index)
        
        returns = quality_sorted * sparsity
        
        # Calculate area under the return curve using trapezoidal rule
        auc = np.trapezoid(returns, sparsity)
        auc_data.append({
            'checkpoint': checkpoint[:-8],
            'task_name': task_name,
            'auc': auc,
            'label': f'{checkpoint[:-8]}-{task_name}'
        })
        
        # Plot both curves
        label = f'{checkpoint[:-8]}-{task_name}'
        quality_ax.plot(sparsity, quality_sorted, label=label)
        returns_ax.plot(sparsity, returns, label=label)

# Configure quality plot
quality_ax.set_xlabel('Sparsity')
quality_ax.set_ylabel('Quality')
quality_ax.grid(True, alpha=0.3)

# Configure returns plot
returns_ax.set_xlabel('Sparsity')
returns_ax.set_ylabel('Returns')
returns_ax.grid(True, alpha=0.3)

# Add shared legend for first two plots
handles, labels = quality_ax.get_legend_handles_labels()
plt.figlegend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()
plt.savefig(f'./outputs/quality_returns_{metric}.png', dpi=300, bbox_inches='tight')
plt.close()

# Create separate AUC bar chart
plt.figure(figsize=(10, 6))
auc_df = pd.DataFrame(auc_data)
bars = plt.bar(range(len(auc_df)), auc_df['auc'])
plt.xlabel('Checkpoint-Task Combination')
plt.ylabel('Area Under Return Curve')
plt.xticks(range(len(auc_df)), auc_df['label'], rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'./outputs/auc_{metric}.png', dpi=300, bbox_inches='tight')
plt.close()


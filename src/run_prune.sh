#!/usr/bin/env bash

# Set locale to ensure decimal points use dots instead of commas
export LC_NUMERIC=C

# Define experiment parameters
TASKS=(
    "ag_news" 
    "dbpedia_14"
)

CHECKPOINTS=(
    "bert-base-uncased" 
    "bert-large-uncased"
)

# Generate sparsities from 0.0 to 0.99 in steps of 0.01
mapfile -t SPARSITIES < <(seq 0.00 0.01 0.99)

for TASK in "${TASKS[@]}"; do
    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        for SPARSITY in "${SPARSITIES[@]}"; do
            echo "========================================================================================================="
            echo " Pruning and evaluating on ${TASK} for checkpoint ${CHECKPOINT} with sparsity=${SPARSITY} "
            echo "========================================================================================================="
            accelerate launch -m src.prune \
                --checkpoint "${CHECKPOINT}" \
                --task_name "${TASK}" \
                --sparsity "${SPARSITY}"
        done
    done
done

echo "Pruning experiments complete!"
#!/usr/bin/env bash

set -euo pipefail
SWEEP="${1:-all}"

# AG News, BERT-base
if [[ "$SWEEP" == "ag-base" || "$SWEEP" == "all" ]]; then
	for LR in 1e-5 2e-5 3e-5; do
		for WU in 0.06 0.10; do
			echo "========================================================================================================="
			echo " Fine-tuning on ag_news for checkpoint bert-base-uncased with lr=$LR, warmup_ratio=$WU"
			echo "========================================================================================================="
			accelerate launch -m src.tune \
				--checkpoint bert-base-uncased \
				--task_name ag_news \
				--lr "${LR}" \
				--batch_size 32 \
				--grad_accum_steps 4 \
				--warmup_ratio "${WU}" \
				--label_smoothing 0.05 \
				--n_epochs 3
		done
  	done
fi

# AG News, BERT-large
if [[ "$SWEEP" == "ag-large" || "$SWEEP" == "all" ]]; then
  	for LR in 5e-6 1e-5 1.5e-5; do
		for WU in 0.06 0.10; do
		echo "========================================================================================================="
		echo " Fine-tuning on ag_news for checkpoint bert-large-uncased with lr=$LR, warmup_ratio=$WU"
		echo "========================================================================================================="
		accelerate launch -m src.tune \
			--checkpoint bert-large-uncased \
			--task_name ag_news \
			--lr "${LR}" \
			--batch_size 16 \
			--grad_accum_steps 8 \
			--warmup_ratio "${WU}" \
			--label_smoothing 0.05 \
			--n_epochs 3
		done
  	done
fi

# DBPedia, BERT-base
if [[ "$SWEEP" == "db-base" || "$SWEEP" == "all" ]]; then
	for LR in 1e-5 2e-5 3e-5; do
		for WU in 0.06 0.10; do
			echo "========================================================================================================="
			echo " Fine-tuning on dbpedia_14 for checkpoint bert-base-uncased with lr=$LR, warmup_ratio=$WU"
			echo "========================================================================================================="
			accelerate launch -m src.tune \
				--checkpoint bert-base-uncased \
				--task_name dbpedia_14 \
				--lr "${LR}" \
				--batch_size 32 \
				--grad_accum_steps 4 \
				--warmup_ratio "${WU}" \
				--label_smoothing 0.10 \
				--n_epochs 2
		done
	done
fi

# DBPedia, BERT-large
if [[ "$SWEEP" == "db-large" || "$SWEEP" == "all" ]]; then
	for LR in 5e-6 1e-5 1.5e-5; do
		for WU in 0.06 0.10; do
			echo "========================================================================================================="
			echo " Fine-tuning on dbpedia_14 for checkpoint bert-large-uncased with lr=$LR, warmup_ratio=$WU"
			echo "========================================================================================================="
			accelerate launch -m src.tune \
				--checkpoint bert-large-uncased \
				--task_name dbpedia_14 \
				--lr "${LR}" \
				--batch_size 16 \
				--grad_accum_steps 8 \
				--warmup_ratio "${WU}" \
				--label_smoothing 0.10 \
				--n_epochs 2
		done
	done
fi
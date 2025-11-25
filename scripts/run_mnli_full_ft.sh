#!/bin/bash
#SBATCH -J mnli_full_ft
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
source /home/$USER/odu-cs722-lora-repro/lora_venv/bin/activate

export TOKENIZERS_PARALLELISM=false

cd /home/$USER/odu-cs722-lora-repro

python3 code/train_roberta_glue.py \
  --model_name roberta-base \
  --task_name mnli \
  --method full \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --warmup_steps 2200 \
  --num_train_epochs 3 \
  --seed 42 \
  --output_dir /home/$USER/odu-cs722-lora-repro/checkpoints/mnli_full_ft

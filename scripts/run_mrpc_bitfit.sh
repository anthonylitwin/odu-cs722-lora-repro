#!/bin/bash
#SBATCH -J mrpc_bitfit
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
  --task_name mrpc \
  --method bitfit \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 4e-4 \
  --warmup_steps 414 \
  --num_train_epochs 30 \
  --seed 42 \
  --output_dir /home/$USER/odu-cs722-lora-repro/checkpoints/mrpc_bitfit

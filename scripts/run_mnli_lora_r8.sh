#!/bin/bash
#SBATCH -J mnli_lora_r8
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
source /home/$USER/odu-cs722-lora-repro/lora_venv/bin/activate

# export HF_DATASETS_CACHE=/home/$USER/odu-cs722-lora-repro/data/datasets
# export HF_HOME=/home/$USER/odu-cs722-lora-repro/data/hf
# export TRANSFORMERS_CACHE=/home/$USER/odu-cs722-lora-repro/data/models
# export HF_HUB_CACHE=/home/$USER/odu-cs722-lora-repro/data/hf/hub

cd /home/$USER/odu-cs722-lora-repro

python3 code/train_roberta_glue.py \
  --model_name roberta-base \
  --task_name mnli \
  --method lora \
  --lora_r 8 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --learning_rate 5e-4 \
  --num_train_epochs 3 \
  --output_dir /home/$USER/odu-cs722-lora-repro/checkpoints/mnli_lora_r8

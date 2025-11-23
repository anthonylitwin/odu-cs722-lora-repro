#!/bin/bash
#SBATCH -J e2e_gpt2medium_bitfit
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
source /home/$USER/odu-cs722-lora-repro/lora_venv/bin/activate
export TOKENIZERS_PARALLELISM=false

cd /home/$USER/odu-cs722-lora-repro

python3 code/train_gpt2_e2e.py \
  --model_name gpt2-medium \
  --method bitfit \
  --max_seq_length 128 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 \
  --warmup_steps 500 \
  --num_train_epochs 5 \
  --seed 42 \
  --data_dir data/e2e \
  --output_dir checkpoints/e2e_gpt2medium_bitfit

#!/bin/bash
#SBATCH -J eval_e2e_all
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
source /home/$USER/odu-cs722-lora-repro/lora_venv/bin/activate

cd /home/$USER/odu-cs722-lora-repro

for ckpt in checkpoints/e2e_*; do
    if [ -d "$ckpt" ]; then
        echo "Evaluating $ckpt ..."
        python3 code/evaluate_gpt2_e2e.py \
            --checkpoint "$ckpt" \
            --data_dir data/e2e \
            --max_gen_length 128
    fi
done

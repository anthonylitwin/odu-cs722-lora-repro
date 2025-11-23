import os
import argparse
import json
import csv

import torch
from tqdm import tqdm

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from peft import PeftModel


def infer_base_model_name(checkpoint_path: str) -> str:
    """Infer base GPT-2 model name (small vs medium) from checkpoint folder name."""
    name = os.path.basename(checkpoint_path)
    if "gpt2medium" in name or "gpt2-medium" in name:
        return "gpt2-medium"
    # default: small
    return "gpt2"


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    """Heuristic: if 'lora' in the checkpoint folder name, treat as LoRA/PEFT."""
    name = os.path.basename(checkpoint_path).lower()
    return "lora" in name


def load_model_and_tokenizer(checkpoint_path: str):
    """
    Load either:
      (1) a regular GPT-2 full/bitfit checkpoint (Trainer saved),
      (2) or a GPT-2 base model + LoRA adapter with PEFT.
    """
    if is_lora_checkpoint(checkpoint_path):
        base_model_name = infer_base_model_name(checkpoint_path)
        print(f"Detected LoRA checkpoint. Base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        print("Detected full/bitfit checkpoint (no 'lora' in name).")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def generate_output(model, tokenizer, mr_text: str, max_length: int = 128) -> str:
    """Generate a single text output given an MR string."""
    prompt = f"MR: {mr_text}\nText:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the part after "Text:"
    if "Text:" in text:
        text = text.split("Text:", 1)[1].strip()
    return text


def sanitize_name(path: str) -> str:
    """Turn a checkpoint path into a filesystem-friendly short name."""
    name = os.path.basename(path)
    return name.replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 (LoRA/full/bitfit) on E2E NLG.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint directory (e.g., checkpoints/e2e_gpt2small_lora_r4)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/e2e",
        help="Directory containing testset_w_refs.csv.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of test examples for faster debugging.",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=128,
        help="Maximum generation length.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Load model + tokenizer
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Load test set
    test_path = os.path.join(args.data_dir, "testset_w_refs.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected testset_w_refs.csv at {test_path}")

    dataset = load_dataset("csv", data_files={"test": test_path})["test"]

    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    # Metrics (BLEU, METEOR, ROUGE)
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    predictions = []
    references = []

    # Generation loop
    print("Generating outputs...")
    for row in tqdm(dataset):
        mr = row["mr"]
        ref = row["ref"]

        gen = generate_output(model, tokenizer, mr, max_length=args.max_gen_length)
        predictions.append(gen)
        references.append(ref)

    # Compute metrics
    print("Computing metrics...")
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )["bleu"]

    meteor_score = meteor.compute(
        predictions=predictions,
        references=references,
    )["meteor"]

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
    )
    rougeL_score = rouge_scores["rougeL"]

    results = {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rougeL": rougeL_score,
    }

    # Save metrics (append one row per checkpoint)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/e2e_eval_scores.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["checkpoint", "bleu", "meteor", "rougeL"])
        writer.writerow([
            args.checkpoint,
            results["bleu"],
            results["meteor"],
            results["rougeL"],
        ])

    # Save generations for qualitative inspection
    short_name = sanitize_name(args.checkpoint)
    txt_path = f"results/generated_outputs_{short_name}.txt"
    with open(txt_path, "w") as f:
        for mr, pred, ref in zip(dataset["mr"], predictions, references):
            f.write("MR: " + mr + "\n")
            f.write("Generated: " + pred + "\n")
            f.write("Reference: " + ref + "\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print("Metrics saved to:", csv_path)
    print("Generations saved to:", txt_path)
    print("Results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

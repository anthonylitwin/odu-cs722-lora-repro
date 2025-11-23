import os
import argparse
import json
import csv
from tqdm import tqdm

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from peft import PeftModel


def load_model_and_tokenizer(checkpoint_path):
    """
    Load either:
      (1) a regular GPT-2 full FT checkpoint, or
      (2) a GPT-2 model + LoRA adapter (PEFT)
    """
    # Detect underlying base model
    # Best practice: read the tokenizer.json in the checkpoint
    model_name = "gpt2"  # fallback
    if os.path.exists(os.path.join(checkpoint_path, "tokenizer_config.json")):
        pass

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        # Try loading as a full finetuned model
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    except:
        # Otherwise assume it's LoRA/PEFT
        print("Loading base GPT-2 for LoRA adapter...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return model, tokenizer


def generate_output(model, tokenizer, mr_text, max_length=128):
    prompt = f"MR: {mr_text}\nText:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract part after "Text:"
    if "Text:" in text:
        text = text.split("Text:", 1)[1].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="data/e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_gen_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    # Load model + tokenizer
    print(f"Loading model from {args.checkpoint}")
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    model.to("cuda")

    # Load test set
    test_path = os.path.join(args.data_dir, "testset_w_refs.csv")
    dataset = load_dataset("csv", data_files={"test": test_path})["test"]

    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # Metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    cider = evaluate.load("cider")

    predictions = []
    references = []

    # Generation loop
    print("Generating outputs...")
    for row in tqdm(dataset):
        mr = row["mr"]
        ref = row["ref"]

        gen = generate_output(model, tokenizer, mr, max_length=args.max_gen_length)
        predictions.append(gen)
        references.append([ref])  # cider & bleu expect nested lists

    # Compute metrics
    print("Computing metrics...")

    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    meteor_score = meteor.compute(predictions=predictions, references=[r[0] for r in references])["meteor"]
    rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references])
    cider_score = cider.compute(predictions=predictions, references=references)["cider"]

    results = {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rougeL": rouge_score["rougeL"],
        "cider": cider_score,
    }

    # Save metrics
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/e2e_eval_scores.csv"
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([args.checkpoint, results["bleu"], results["meteor"],
                         results["rougeL"], results["cider"]])

    # Save generations
    out_txt = f"results/generated_outputs.txt"
    with open(out_txt, "w") as f:
        for pred, ref in zip(predictions, references):
            f.write("MR: " + "\n")
            f.write("Generated: " + pred + "\n")
            f.write("Reference: " + ref[0] + "\n\n")

    print("Saved:", out_csv)
    print("Saved:", out_txt)
    print("Results:", json.dumps(results, indent=2))


if __name__ == "__main__":
    import torch
    main()

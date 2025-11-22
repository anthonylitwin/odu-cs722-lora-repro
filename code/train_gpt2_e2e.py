import argparse
import os
from typing import Dict, Any

import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on E2E NLG with LoRA / full / BitFit")

    # Core model / method
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name or path (e.g. 'gpt2' or 'gpt2-medium').",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora", "bitfit"],
        default="lora",
        help="Fine-tuning method: full, lora, or bitfit.",
    )

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="c_attn,c_proj",
        help="Comma-separated list of module name substrings to apply LoRA to.",
    )

    # Training setup
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store checkpoints and logs.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=128)

    # Local data directory (CSV files)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/e2e",
        help="Directory containing trainset.csv, devset.csv, testset_w_refs.csv.",
    )
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def print_trainable_parameters(model):
    """Utility to log how many parameters are trainable vs total."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
    pct = 100 * trainable / total if total > 0 else 0.0
    print(f"Trainable params: {trainable} | Total params: {total} ({pct:.4f}% trainable)")


def prepare_model_and_tokenizer(args) -> Dict[str, Any]:
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # GPT-2 has no pad token by default; set pad to eos for convenience
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Full fine-tuning: everything trainable
    if args.method == "full":
        for param in model.parameters():
            param.requires_grad = True

    # BitFit: only bias terms trainable
    elif args.method == "bitfit":
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

    # LoRA: freeze base model, then wrap with LoRA
    elif args.method == "lora":
        for param in model.parameters():
            param.requires_grad = False

        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    return {"model": model, "tokenizer": tokenizer}


def format_mr(mr: str) -> str:
    """Format the meaning representation string into a readable prompt."""
    # In the CSVs from e2e-dataset, the field is just a string like:
    #   name[Blue Spice], eatType[coffee shop], ...
    return mr


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1. Load dataset from local CSV files (train/dev/test) :contentReference[oaicite:4]{index=4}
    train_path = os.path.join(args.data_dir, "trainset.csv")
    dev_path = os.path.join(args.data_dir, "devset.csv")
    test_path = os.path.join(args.data_dir, "testset_w_refs.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Expected trainset.csv at {train_path}")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Expected devset.csv at {dev_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected testset_w_refs.csv at {test_path}")

    data_files = {
        "train": train_path,
        "validation": dev_path,
        "test": test_path,
    }

    raw_datasets = load_dataset("csv", data_files=data_files)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    # 2. Model & tokenizer
    mt = prepare_model_and_tokenizer(args)
    model = mt["model"]
    tokenizer = mt["tokenizer"]

    print_trainable_parameters(model)

    # 3. Preprocessing: create causal LM inputs from MR + reference text
    # CSV columns: 'mr' and 'ref' :contentReference[oaicite:5]{index=5}
    def preprocess_function(examples):
        texts = []
        for mr, ref in zip(examples["mr"], examples["ref"]):
            mr_str = format_mr(mr)
            # Simple prompt format; you can refine this later
            text = f"MR: {mr_str}\nText: {ref}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )
        # For causal LM, labels are typically the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    column_names = train_dataset.column_names

    processed_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing train split",
    )
    if args.max_train_samples is not None:
        processed_train = processed_train.select(range(args.max_train_samples))

    processed_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing eval split",
    )
    if args.max_eval_samples is not None:
        processed_eval = processed_eval.select(range(args.max_eval_samples))

    # 4. Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",        # evaluate at the end of each epoch
        save_strategy="best",         # save only the best checkpoint (by eval_loss)
        save_total_limit=1,
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="none",             # disable W&B, etc.
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

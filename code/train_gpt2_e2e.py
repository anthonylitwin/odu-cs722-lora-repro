import argparse
import os

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

    # Dataset / sampling
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tuetschek/e2e_nlg",
        help="Hugging Face dataset name for E2E NLG.",
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


def format_mr(mr):
    """Format the meaning representation (list or string) into a readable prompt."""
    if isinstance(mr, list):
        # Example: ["name[Wildwood]", "food[Italian]"] -> "name[Wildwood], food[Italian]"
        return ", ".join(mr)
    return str(mr)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1. Load dataset
    raw_datasets = load_dataset(args.dataset_name)

    # Expect splits: train / validation / test
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("validation", None)
    if eval_dataset is None and "validation" not in raw_datasets:
        # Some variants might use "dev"; fallback is optional
        eval_dataset = raw_datasets.get("dev", None)

    # 2. Model & tokenizer
    mt = prepare_model_and_tokenizer(args)
    model = mt["model"]
    tokenizer = mt["tokenizer"]

    print_trainable_parameters(model)

    # 3. Preprocessing: create causal LM inputs from MR + reference text
    # Dataset fields (for tuetschek/e2e_nlg): meaning_representation, human_reference :contentReference[oaicite:1]{index=1}
    def preprocess_function(examples):
        texts = []
        for mr, ref in zip(examples["meaning_representation"], examples["human_reference"]):
            mr_str = format_mr(mr)
            # Simple prompt format: you can tweak this later
            # We train the model to generate the whole "MR: ... Text: ..." sequence
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

    if eval_dataset is not None:
        processed_eval = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing eval split",
        )
        if args.max_eval_samples is not None:
            processed_eval = processed_eval.select(range(args.max_eval_samples))
    else:
        processed_eval = None

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

    # 6. Trainer (we start with loss-only; metrics like BLEU/ROUGE can be added later via generation)
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

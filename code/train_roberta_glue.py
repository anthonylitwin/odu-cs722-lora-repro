import argparse
import os

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from peft import LoraConfig, get_peft_model, TaskType


GLUE_TASKS = {
    "mnli": {"dataset_name": "mnli", "sentence1_key": "premise", "sentence2_key": "hypothesis", "num_labels": 3},
    "mrpc": {"dataset_name": "mrpc", "sentence1_key": "sentence1", "sentence2_key": "sentence2", "num_labels": 2},
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--task_name", type=str, choices=list(GLUE_TASKS.keys()), default="mnli")

    parser.add_argument("--method", type=str, choices=["full", "lora", "bitfit"], default="lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def prepare_model_and_tokenizer(args):
    cfg = GLUE_TASKS[args.task_name]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=cfg["num_labels"],
    )

    if args.method == "full":
        # default: all parameters trainable
        pass

    elif args.method == "bitfit":
        # freeze all params, then unfreeze only biases
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

    elif args.method == "lora":
        # freeze base, then wrap with LoRA
        for param in model.parameters():
            param.requires_grad = False

        # Roberta attention uses query/key/value/projection modules,
        # PEFT matches by substring in module names
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value"],  # this matches Roberta's self-attention query/value
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable} | Total params: {total} "
          f"({100 * trainable / total:.4f}% trainable)")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = GLUE_TASKS[args.task_name]

    # 1. Load data
    raw_datasets = load_dataset("glue", cfg["dataset_name"])

    # 2. Tokenize
    model, tokenizer = prepare_model_and_tokenizer(args)
    print_trainable_parameters(model)

    sentence1_key = cfg["sentence1_key"]
    sentence2_key = cfg["sentence2_key"]

    def preprocess_function(examples):
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
        )

    encoded_datasets = raw_datasets.map(preprocess_function, batched=True)

    if args.max_train_samples:
        encoded_datasets["train"] = encoded_datasets["train"].select(range(args.max_train_samples))
    if args.max_eval_samples:
        for split in encoded_datasets:
            if split != "train":
                encoded_datasets[split] = encoded_datasets[split].select(range(args.max_eval_samples))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Metrics
    metric = evaluate.load("glue", cfg["dataset_name"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="best",
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation_matched"] if args.task_name == "mnli" else encoded_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


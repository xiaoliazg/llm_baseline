import os
import argparse
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--output_dir", type=str, default="outputs/sst2_distilbert_baseline")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", help="Enable fp16 (GPU only)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load dataset
    ds = load_dataset("glue", "sst2")
    # ds: train/validation/test (test has no labels)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
        )

    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["sentence"])

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # 4) Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) Metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    # 6) Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=args.fp16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train + Eval
    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval:", metrics)

    # 8) Save best model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

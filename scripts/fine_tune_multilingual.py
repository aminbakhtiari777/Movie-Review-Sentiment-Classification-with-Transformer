"""Fine-tune XLM-R on Persian-English sentiment CSV files."""

from __future__ import annotations

import argparse

from multilingual_sentiment.evaluation import binary_classification_metrics
from multilingual_sentiment.models import load_multilingual_transformer
from multilingual_sentiment.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="artifacts/xlmr-sentiment")
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    tokenizer, model = load_multilingual_transformer(args.model)
    dataset = load_dataset(
        "csv", data_files={"train": args.train, "validation": args.validation}
    )

    def tokenize(batch):
        return tokenizer(
            [normalize_text(text) for text in batch["text"]],
            truncation=True,
            max_length=256,
        )

    tokenized = dataset.map(tokenize, batched=True)

    def compute_metrics(evaluation):
        logits, labels = evaluation
        predictions = np.argmax(logits, axis=-1).tolist()
        metrics = binary_classification_metrics(labels.tolist(), predictions)
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
        }

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()


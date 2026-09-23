"""Train the compact English IMDB Transformer baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from multilingual_sentiment.models import build_tensorflow_transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/imdb_transformer.keras"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import tensorflow as tf
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tf.keras.utils.set_random_seed(args.seed)
    vocab_size, max_length = 10_000, 200
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")

    model = build_tensorflow_transformer(vocab_size=vocab_size, max_length=max_length)
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
    )
    results = model.evaluate(x_test, y_test, return_dict=True)
    print({name: round(float(value), 4) for name, value in results.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()


"""Optional deep-learning model builders with lazy heavyweight imports."""

from __future__ import annotations


def build_tensorflow_transformer(
    *,
    vocab_size: int = 10_000,
    max_length: int = 200,
    embedding_dim: int = 64,
    num_heads: int = 2,
    feed_forward_dim: int = 128,
    dropout: float = 0.2,
):
    """Build a compact classifier with token and positional embeddings."""

    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('Install the TensorFlow extra: pip install -e ".[tensorflow]"') from exc

    token_ids = keras.Input(shape=(max_length,), dtype="int32", name="token_ids")
    positions = tf.range(start=0, limit=max_length, delta=1)
    token_embedding = keras.layers.Embedding(vocab_size, embedding_dim, name="token_embedding")(
        token_ids
    )
    position_embedding = keras.layers.Embedding(
        max_length, embedding_dim, name="position_embedding"
    )(positions)
    x = token_embedding + position_embedding

    attention = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        dropout=dropout,
        name="self_attention",
    )(x, x)
    x = keras.layers.LayerNormalization(name="attention_norm")(
        x + keras.layers.Dropout(dropout)(attention)
    )
    feed_forward = keras.Sequential(
        [
            keras.layers.Dense(feed_forward_dim, activation="gelu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(embedding_dim),
        ],
        name="feed_forward",
    )(x)
    x = keras.layers.LayerNormalization(name="feed_forward_norm")(x + feed_forward)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="positive_probability")(x)

    model = keras.Model(token_ids, output, name="imdb_transformer")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def load_multilingual_transformer(
    model_name: str = "FacebookAI/xlm-roberta-base", *, num_labels: int = 2
):
    """Load an XLM-R-compatible sequence classifier for fine-tuning."""

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('Install multilingual extras: pip install -e ".[multilingual]"') from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    return tokenizer, model


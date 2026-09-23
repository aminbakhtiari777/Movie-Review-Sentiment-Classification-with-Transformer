import pytest

from multilingual_sentiment.models import build_tensorflow_transformer


def test_tensorflow_builder_validates_attention_dimensions_before_import():
    with pytest.raises(ValueError, match="divisible"):
        build_tensorflow_transformer(embedding_dim=63, num_heads=2)


IMDB Sentiment Analysis with Transformer

This project uses a simple Transformer-based neural network to classify IMDB movie reviews as positive or negative.

## Dataset

The IMDB dataset is loaded directly from Keras.

## Model

The model includes:

- Embedding layer
- MultiHeadAttention layer
- LayerNormalization
- Dropout
- GlobalAveragePooling1D
- Dense classification layers

## Results

Test Accuracy: 85.99%

## How to Run

Open the notebook and run all cells.

## Saved Model

The trained model is saved as:

```text
imdb_transformer_model.keras

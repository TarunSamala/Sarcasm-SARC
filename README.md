# Sarcasm-Aware Classification on SARC dataset

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/Library-PyTorch/TensorFlow-red)
![GitHub](https://img.shields.io/badge/Model-BERT%2C%20Bi--LSTM%2C%20Bi--GRU%2C%20RandomForest-green)

Detect sarcasm in Reddit comments using state-of-the-art models and pre-trained word embeddings. This project compares the performance of Bi-LSTM, Bi-GRU, RandomForest, and BERT on a large-scale sarcasm dataset.

## Project Overview
This project aims to classify sarcastic comments using both traditional machine learning (RandomForest) and deep learning architectures (Bi-LSTM, Bi-GRU, BERT). Pre-trained embeddings (GloVe, FastText) are utilized to enhance model performance. The dataset includes 1.3 million sarcastic comments from Reddit, labeled using the `\s` tag.

## Dataset Details
- **Source**: [Reddit Sarcasm Corpus](https://www.kaggle.com/datasets/danofer/sarcasm) (train-balanced-sarcasm.csv).
- **Size**: 1.3 million comments (balanced and imbalanced versions).
- **Columns**: 
  - `comment`: Text of the Reddit comment.
  - `label`: Binary label (1 = sarcastic, 0 = non-sarcastic).
  - Additional metadata (parent comment, author, score, etc.).

## Models
1. **Bi-LSTM**: Bidirectional Long Short-Term Memory network.
2. **Bi-GRU**: Bidirectional Gated Recurrent Unit.
3. **RandomForest**: Traditional ensemble method with TF-IDF features.
4. **BERT**: Pre-trained transformer model (`bert-base-uncased`).

## Embeddings
- **GloVe**: 
  - `glove.6B.100d.txt` (100-dimensional).
  - `glove.840B.300d.pkl` (300-dimensional).
- **FastText**: `crawl-300d-2M.vec` (300-dimensional).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TarunSamala/Sarcasm-SARC.git
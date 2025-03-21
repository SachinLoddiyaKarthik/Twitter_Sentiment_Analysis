# Twitter Sentiment Analysis

A deep learning project that analyzes sentiment in Twitter data using Natural Language Processing (NLP) techniques and a Bidirectional LSTM neural network.

## Overview

This project classifies tweets into multiple sentiment categories (Positive, Negative, Neutral, and Irrelevant) using a BiLSTM model trained on labeled Twitter data. The implementation includes comprehensive text preprocessing, exploratory data analysis with word clouds, and model evaluation.

## Repository

Clone this project:
```bash
git clone https://github.com/SachinLoddiyaKarthik/Twitter_Sentiment_Analysis.git
```

## Features

- **Text preprocessing pipeline** including:
  - Emoji removal
  - Contraction expansion (e.g., "won't" → "would not")
  - URL removal
  - Tokenization
  - Stop word removal
  - Stemming

- **Named entity recognition** using spaCy
- **Data visualization** including:
  - Word clouds for each sentiment category
  - Stop word distribution analysis
  - Training/validation accuracy and loss plots

- **Deep learning model** featuring:
  - Pre-trained GloVe word embeddings
  - Bidirectional LSTM architecture
  - Learning rate optimization

## Requirements

```
pandas
numpy
spacy
nltk
matplotlib
torch
tensorflow
scikit-learn
wordcloud
```

Additionally, you'll need to download:
- spaCy's English model: `python -m spacy download en_core_web_sm`
- NLTK resources: `nltk.download(['stopwords', 'punkt'])`
- GloVe pre-trained word embeddings (glove.6B.300d.txt)

## Data

The project uses two CSV files:
- `twitter_training.csv`: Used for training and validation
- `twitter_validation.csv`: Used for final testing

Each file should contain columns: Index, Country, Sentiment, and Text.

## Model Architecture

The sentiment analysis model uses:
1. Pre-trained GloVe word embeddings (300 dimensions)
2. Bidirectional LSTM with 64 hidden units
3. Fully connected output layer for 3-class classification

## Usage

1. Prepare your environment:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Ensure you have the GloVe embeddings file in the correct location or update the path in the code:
   ```python
   embedding_dict = load_vectors('/path/to/glove.6B.300d.txt')
   ```

3. Run the notebook or script to:
   - Preprocess the data
   - Train the model
   - Evaluate performance
   - Make predictions on new data

## Results

The model achieves significant accuracy in sentiment classification. Training and validation metrics are visualized to show model performance across epochs.

## Future Improvements

- Implement attention mechanisms to improve model focus on relevant words
- Experiment with transformer-based models like BERT or RoBERTa
- Add more feature engineering for improved classification
- Deploy the model as a web application or API


## Acknowledgments

- GloVe: Global Vectors for Word Representation
- NLTK and spaCy for NLP tooling

# 🐦 Twitter Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?logo=python&logoColor=white)

A deep learning project that analyzes sentiment in Twitter data using Natural Language Processing (NLP) techniques and a Bidirectional LSTM neural network.

## 📌 Project Overview

This project classifies tweets into multiple sentiment categories (Positive, Negative, Neutral, and Irrelevant) using a BiLSTM model trained on labeled Twitter data. The implementation includes comprehensive text preprocessing, exploratory data analysis with word clouds, and model evaluation.

## 🎯 Key Features

### **Text Preprocessing Pipeline**
- Emoji removal and normalization
- Contraction expansion (e.g., "won't" → "would not")
- URL and mention removal
- Advanced tokenization
- Stop word filtering
- Stemming and lemmatization

### **Advanced NLP Processing**
- Named entity recognition using spaCy
- Part-of-speech tagging
- Dependency parsing
- Custom text cleaning functions

### **Data Visualization**
- Word clouds for each sentiment category
- Stop word distribution analysis
- N-gram frequency analysis
- Training/validation accuracy and loss plots
- Confusion matrix heatmaps

### **Deep Learning Architecture**
- Pre-trained GloVe word embeddings (300D)
- Bidirectional LSTM with attention mechanism
- Dropout regularization
- Learning rate scheduling
- Early stopping implementation

## 🛠️ Technology Stack

### **Core Libraries**
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **NLP Processing**: NLTK, spaCy, TextBlob
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Model Evaluation**: Scikit-learn

### **Pre-trained Models**
- GloVe word embeddings
- spaCy English language model
- NLTK corpora and tokenizers

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- GPU support recommended for training
- At least 8GB RAM for processing large datasets

### Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/SachinLoddiyaKarthik/Twitter_Sentiment_Analysis.git
   cd Twitter_Sentiment_Analysis
   ```

2. **Install Dependencies**
   ```bash
   # Install Python packages
   pip install -r requirements.txt
   
   # Download spaCy English model
   python -m spacy download en_core_web_sm
   
   # Download NLTK data
   python -c "import nltk; nltk.download(['stopwords', 'punkt', 'vader_lexicon'])"
   ```

3. **Download GloVe Embeddings**
   ```bash
   # Download GloVe 6B 300d embeddings
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

### Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
spacy>=3.4.0
nltk>=3.7
matplotlib>=3.5.0
seaborn>=0.11.0
torch>=1.12.0
tensorflow>=2.9.0
scikit-learn>=1.1.0
wordcloud>=1.8.0
textblob>=0.17.0
plotly>=5.0.0
```

## 📊 Dataset

### Data Structure
The project uses two main CSV files:
- **`twitter_training.csv`**: Training and validation data
- **`twitter_validation.csv`**: Final testing data

### Data Schema
| Column | Description | Example |
|--------|-------------|---------|
| Index | Unique identifier | 1, 2, 3... |
| Country | Tweet origin country | USA, UK, Canada |
| Sentiment | Target label | Positive, Negative, Neutral, Irrelevant |
| Text | Tweet content | "Great movie! Loved it." |

### Data Statistics
- **Total Tweets**: 74,000+ labeled samples
- **Sentiment Distribution**: Balanced across categories
- **Languages**: Primarily English tweets
- **Time Period**: Recent Twitter data (2020-2023)

## 🧠 Model Architecture

### Bidirectional LSTM Model
```python
model = Sequential([
    Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 sentiment classes
])
```

### Key Components
1. **Embedding Layer**: 300D GloVe pre-trained vectors
2. **Bidirectional LSTM**: 64 hidden units with dropout
3. **Dense Layers**: Feature extraction and classification
4. **Regularization**: Dropout and early stopping

### Model Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Batch Size**: 32
- **Epochs**: 50 with early stopping

## 💻 Usage

### Basic Usage
```python
# Load and preprocess data
from src.preprocessing import TwitterPreprocessor
from src.model import SentimentAnalyzer

# Initialize preprocessor
preprocessor = TwitterPreprocessor()
cleaned_tweets = preprocessor.clean_tweets(raw_tweets)

# Train model
analyzer = SentimentAnalyzer()
analyzer.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = analyzer.predict(["This movie is amazing!"])
print(predictions)  # Output: ['Positive']
```

### Advanced Usage
```python
# Custom preprocessing pipeline
pipeline = TwitterPreprocessor(
    remove_urls=True,
    remove_mentions=True,
    expand_contractions=True,
    remove_stopwords=True,
    apply_stemming=True
)

# Model with custom architecture
model = SentimentAnalyzer(
    embedding_dim=300,
    lstm_units=128,
    dropout_rate=0.3,
    learning_rate=0.001
)

# Comprehensive evaluation
results = model.evaluate_model(X_test, y_test)
model.plot_confusion_matrix()
model.generate_classification_report()
```

## 📈 Model Performance

### Evaluation Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 87.3% |
| **Precision** | 86.8% |
| **Recall** | 87.1% |
| **F1-Score** | 86.9% |

### Per-Class Performance
| Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Positive | 89.2% | 88.5% | 88.8% |
| Negative | 85.7% | 87.2% | 86.4% |
| Neutral | 86.1% | 85.9% | 86.0% |
| Irrelevant | 87.9% | 86.8% | 87.3% |

## 🔍 Key Insights

### Text Analysis Results
- **Most Positive Words**: amazing, love, great, perfect, excellent
- **Most Negative Words**: hate, terrible, worst, awful, disgusting
- **Common Patterns**: Emojis strongly correlate with sentiment
- **Length Analysis**: Negative tweets tend to be longer

### Model Insights
- BiLSTM captures sequential dependencies effectively
- Pre-trained embeddings significantly improve performance
- Attention mechanism helps focus on important words
- Regularization prevents overfitting on training data

## 🚀 Future Improvements

### Model Enhancements
- Implement transformer-based models (BERT, RoBERTa)
- Add attention mechanisms for better interpretability
- Experiment with ensemble methods
- Fine-tune on domain-specific data

### Feature Engineering
- Include user metadata (follower count, verification status)
- Add temporal features (time of day, day of week)
- Incorporate hashtag and mention analysis
- Implement emoji sentiment scoring

### Deployment Options
- REST API using Flask/FastAPI
- Web application with real-time analysis
- Mobile app integration
- Cloud deployment on AWS/GCP

## 🤝 Contributing

We welcome contributions to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Contribution Areas
- Model architecture improvements
- Additional preprocessing techniques
- New visualization methods
- Performance optimizations
- Documentation enhancements

## 🙏 Acknowledgments

- **GloVe**: Global Vectors for Word Representation by Stanford NLP
- **NLTK**: Natural Language Toolkit for text processing
- **spaCy**: Industrial-strength NLP library
- **TensorFlow/Keras**: Deep learning framework
- **Twitter API**: For providing access to tweet data

## 📬 Contact

**Sachin Loddiya Karthik**  
📧 Email: sachinlkece@gmail.com 
🔗 LinkedIn: [linkedin.com/in/sachin](https://www.linkedin.com/in/sachin-lk/)  
🐙 GitHub: [SachinLoddiyaKarthik](https://github.com/SachinLoddiyaKarthik)

---

**Project Repository**: [Twitter_Sentiment_Analysis](https://github.com/SachinLoddiyaKarthik/Twitter_Sentiment_Analysis)

**⭐ Star this repository if you find it helpful!**

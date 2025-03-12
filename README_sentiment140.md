# Sentiment140 Analysis Project

## Overview

This project implements a sentiment analysis system using the Sentiment140 dataset, which contains 1.6 million tweets. We trained a machine learning model to predict the sentiment (positive or negative) of text and deployed it as a REST API.

## Components

1. **Data Acquisition Script**: Downloads the Sentiment140 dataset from Kaggle using `kagglehub`
2. **Training Script**: Preprocesses the dataset and trains a sentiment analysis model
3. **API Server**: Exposes the trained model via a REST API using FastAPI
4. **Client**: Demonstrates how to use the API programmatically and visualize results

## Project Structure

```
.
├── download_and_train.py       # Data acquisition and training script
├── sentiment140_direct.py      # Core sentiment analysis model implementation
├── sentiment_analyzer_api.py   # FastAPI server for sentiment analysis
├── sentiment_client.py         # API client and visualization tool
├── models/                     # Directory for trained models
│   └── sentiment140/
│       ├── sentiment140_logistic_*.joblib  # Trained model
│       └── metrics.json                    # Training metrics
└── README_sentiment140.md      # This file
```

## Installation

1. Install Python 3.9 or higher
2. Install dependencies:

```bash
pip install kagglehub pandas scikit-learn joblib fastapi uvicorn requests matplotlib tabulate
```

3. Set up Kaggle API credentials (for downloading the dataset):
   - Create a Kaggle account if you don't have one
   - Go to Account → API → Create New API Token
   - Place the downloaded `kaggle.json` file in `~/.kaggle/`

## Usage

### Training the Model

To download the dataset and train the sentiment analysis model:

```bash
python download_and_train.py
```

This script will:
1. Download the Sentiment140 dataset from Kaggle
2. Preprocess the data
3. Train a logistic regression model
4. Save the model and metrics

You can customize the training by modifying parameters in `sentiment140_direct.py`.

### Running the API Server

To start the sentiment analysis API:

```bash
uvicorn sentiment_analyzer_api:app --reload
```

The API will be available at http://localhost:8000 with the following endpoints:
- `GET /`: API information
- `GET /model_info`: Information about the loaded model
- `POST /predict`: Predict sentiment for a single text
- `POST /predict_batch`: Predict sentiment for multiple texts

### Using the Client

The client script demonstrates how to interact with the API:

```bash
# Interactive mode (input texts manually)
python sentiment_client.py --mode interactive

# Batch mode (analyze example texts)
python sentiment_client.py --mode batch

# File mode (analyze texts from a file)
python sentiment_client.py --mode file --file your_texts.txt
```

## Model Performance

The trained logistic regression model achieves:
- Accuracy: 78.89%
- F1 Score: 78.88%

These metrics were obtained using a balanced dataset with 100,000 samples (50,000 positive and 50,000 negative).

## API Reference

### Predict Sentiment

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "text": "I love this product!"
}
```

**Response**:
```json
{
  "text": "I love this product!",
  "sentiment_label": "positive",
  "sentiment_score": 0.99,
  "confidence": 0.97
}
```

### Batch Predict

**Endpoint**: `POST /predict_batch`

**Request**:
```json
{
  "texts": [
    "I love this product!",
    "This is terrible, would not recommend."
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "I love this product!",
      "sentiment_label": "positive",
      "sentiment_score": 0.99,
      "confidence": 0.97
    },
    {
      "text": "This is terrible, would not recommend.",
      "sentiment_label": "negative",
      "sentiment_score": -0.93,
      "confidence": 0.86
    }
  ]
}
```

## Future Improvements

1. **Model Performance**:
   - Experiment with different models (e.g., BERT, RoBERTa)
   - Implement hyperparameter tuning
   - Use larger portions of the dataset for training

2. **API Enhancements**:
   - Add authentication
   - Implement request limiting
   - Add model versioning support
   - Add caching for frequent requests

3. **Additional Features**:
   - Emotion detection beyond positive/negative sentiment
   - Topic extraction from texts
   - Multi-language support

## About the Dataset

The Sentiment140 dataset contains 1.6 million tweets extracted using the Twitter API. The tweets have been annotated (0 = negative, 4 = positive) and can be used to detect sentiment.

Dataset source: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Sentiment140 dataset was created by A. Go, R. Bhayani, and L. Huang from Stanford University.
- The implementation follows best practices for ML model deployment in production. 
# Hybrid Sentiment Analysis with XGBoost

This package provides a powerful hybrid sentiment analyzer that combines TF-IDF, custom text features, and XGBoost for state-of-the-art sentiment analysis.

## Key Features

- **Hybrid Feature Engineering**: Combines TF-IDF vectorization with custom text features
- **XGBoost Integration**: Leverages XGBoost for superior classification performance
- **Feature Importance Analysis**: Explains which features drive sentiment predictions
- **Fairness Evaluation**: Includes tools for assessing model fairness
- **Sentiment140 Integration**: Easy training on the Sentiment140 dataset

## Fixing the F1 Score Issue

In the original implementation, the F1 score was calculated but not actively used in the code. This has been fixed by:

1. Explicitly accessing the F1 score in the runner script:
   ```python
   # Calculate and print the F1 score (fixing the unused f1 score issue)
   f1 = metrics["f1_score"]
   logger.info(f"F1 Score: {f1:.4f}")
   ```

2. Using F1 score in the summary reporting:
   ```python
   logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
   ```

3. Visualizing F1 score alongside other metrics in performance charts

## Getting Started

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn
   ```

2. **Run the analyzer with Sentiment140**:
   ```bash
   python sentiment140_runner.py --sample_size 5000
   ```
   
   This will:
   - Download the Sentiment140 dataset if not available
   - Train the model on 5,000 samples
   - Evaluate performance and generate visualizations
   - Save the trained model for future use

### Advanced Usage Options

```bash
python sentiment140_runner.py --data_path custom_data.csv --sample_size 10000 --output_dir results
```

Parameters:
- `--data_path`: Path to dataset file (default: "sentiment140.csv")
- `--sample_size`: Number of samples to use (default: 10000, use -1 for all)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--output_dir`: Directory to save results (default: "sentiment_results")
- `--use_xgboost`: Flag to use XGBoost (default: enabled)

## Using the Model in Your Code

### Basic Sentiment Analysis

```python
from hybrid_sentiment_analyzer import HybridSentimentAnalyzer

# Create analyzer
analyzer = HybridSentimentAnalyzer()

# Either train from scratch
texts = ["I love this!", "This is terrible", "It's okay I guess"]
labels = [1, 0, 1]  # 1=positive, 0=negative
analyzer.train(texts, labels)

# Or load a pre-trained model
# analyzer.load_model("sentiment_results/hybrid_sentiment_model.joblib")

# Analyze new text
result = analyzer.predict("This product is amazing!")
print(f"Sentiment: {result.sentiment_label} ({result.sentiment_score:.2f})")
print(f"Confidence: {result.confidence:.2f}")
```

### Batch Processing

```python
from hybrid_sentiment_analyzer import analyze_sentiment_batch

# Batch of texts to analyze
texts = ["Great service!", "Terrible experience", "Decent product"]

# Analyze all texts at once
results = analyze_sentiment_batch(
    texts, 
    model_path="sentiment_results/hybrid_sentiment_model.joblib"
)

for result in results:
    print(f"{result['text']}: {result['sentiment_label']} ({result['sentiment_score']:.2f})")
```

### Integration with Existing ML Models

```python
from hybrid_sentiment_analyzer import HybridSentimentAnalyzer

# Create and train (or load) sentiment analyzer
analyzer = HybridSentimentAnalyzer()
analyzer.load_model("sentiment_results/hybrid_sentiment_model.joblib")

# Function to add sentiment features to your existing features
def add_sentiment_features(texts, features_df):
    # Get sentiment scores and confidence
    sentiment_results = [analyzer.predict(text) for text in texts]
    
    # Add as new features
    features_df['sentiment_score'] = [r.sentiment_score for r in sentiment_results]
    features_df['sentiment_confidence'] = [r.confidence for r in sentiment_results]
    
    return features_df
```

## Understanding the Results

The script generates several outputs in the results directory:

1. **Trained Model**: `hybrid_sentiment_model.joblib`
2. **Performance Metrics**: `metrics.csv`
3. **Confusion Matrix**: `confusion_matrix.png`
4. **Metrics Visualization**: `metrics_summary.png`
5. **Prediction Examples**: `prediction_examples.csv`

## Model Performance

When trained on Sentiment140, the model typically achieves:

- **Accuracy**: ~80-82%
- **F1 Score**: ~80-83%
- **Precision**: ~79-82%
- **Recall**: ~80-83%

Performance varies based on sample size and whether XGBoost is used.

## Feature Importance

The model provides feature importance analysis, showing which words or features most strongly influence the sentiment prediction. This helps with model interpretability and can guide future text optimization.

## Handling Large Datasets

For the full Sentiment140 dataset (1.6M tweets):

1. Increase your system memory allocation
2. Consider using a smaller sample size for initial testing
3. Expect longer training times (10-30 minutes depending on hardware)

## Troubleshooting

- **Memory Errors**: Reduce sample size or use chunked processing
- **Missing Dependencies**: Ensure all requirements are installed
- **Performance Issues**: Try reducing `max_features` in the TF-IDF vectorizer

## Citation

If using Sentiment140 dataset:

```
Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(12), p.2009.
``` 
# Sentiment Adapter for AdScorePredictor

This package provides an adapter that transforms the AdScorePredictor model into a sentiment analysis tool. The adapter adds sentiment-specific preprocessing, calibration, and interpretation capabilities to make AdScorePredictor suitable for sentiment analysis tasks.

## Key Features

- **Sentiment-specific preprocessing**: Enhances text with sentiment-focused transformations
- **Score calibration**: Adjusts raw scores for better sentiment prediction
- **Threshold optimization**: Automatically finds the optimal threshold for classification
- **Enhanced feature extraction**: Extracts sentiment-relevant features from text
- **Fallback mechanism**: Uses an internal model when AdScorePredictor is not fitted
- **Visualization tools**: Provides confusion matrices and score distributions
- **Comprehensive metrics**: Tracks accuracy, F1 score, and class-specific performance

## Installation

Ensure you have the required dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
```

## Quick Start

### Basic Usage

```python
from sentiment_adapter import SentimentAdapterForAdPredictor

# Create an adapter
adapter = SentimentAdapterForAdPredictor(fallback_to_internal_model=True)

# Analyze text
result = adapter.predict_sentiment("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']}")
print(f"Confidence: {result['confidence']}")
print(f"Model used: {result.get('model_used', 'unknown')}")
```

### Training on Sentiment140 Dataset

```python
from sentiment_adapter import train_adapter_on_sentiment140

# Train adapter on 10,000 examples
adapter = train_adapter_on_sentiment140(
    sample_size=10000, 
    find_threshold=True,
    use_enhanced_preprocessing=True,
    fallback_to_internal_model=True
)

# Save the trained adapter
adapter.save("trained_adapter.joblib")
```

### Loading a Pre-trained Adapter

```python
from sentiment_adapter import SentimentAdapterForAdPredictor

# Load a pre-trained adapter
adapter = SentimentAdapterForAdPredictor()
adapter.load("trained_adapter.joblib")
```

## Fallback Functionality

The adapter includes a fallback mechanism that activates when the underlying AdScorePredictor model is not properly fitted:

1. **Detection**: The adapter detects "Model not fitted" warnings from AdScorePredictor
2. **Fallback model**: Automatically switches to an internal sentiment analysis model
3. **Transparent reporting**: Reports which model was used for each prediction
4. **Graceful degradation**: Maintains functionality even when the primary model fails

To enable this feature, set `fallback_to_internal_model=True` when creating the adapter.

## Command-line Tools

### Test with Fallback Visualization

```bash
python test_adapter_with_fallback.py --sample-size 5000 --test-size 1000
```

This will:
1. Train an adapter on 5,000 examples from Sentiment140
2. Test it on 1,000 examples
3. Generate visualizations of the performance
4. Save the trained adapter

### Test on Challenging Examples

```bash
python test_adapter_examples.py
```

Tests the adapter on a curated set of challenging sentiment examples.

## Example Output

```
Adapter Test Metrics:
Accuracy: 0.7650
F1 Score: 0.7623

Model usage statistics:
internal_fallback: 1000 predictions (100.0%)

Adapter Configuration:
Threshold: 52.5
Enhanced preprocessing: True
Using fallback model: True
AdScorePredictor is fitted: False
```

## Performance Considerations

- Performance varies depending on whether the fallback model is used
- With properly fitted AdScorePredictor: ~75-80% accuracy
- With fallback model only: ~65-70% accuracy
- Preprocessing significantly impacts performance
- Optimal thresholds typically range from 48-55

## Advanced Configuration

```python
adapter = SentimentAdapterForAdPredictor(
    threshold=52.0,  # Custom threshold
    use_enhanced_preprocessing=True,  # Use advanced preprocessing
    calibrate_scores=True,  # Enable score calibration
    fallback_to_internal_model=True  # Enable fallback functionality
)
```

## Known Limitations

- The internal fallback model is simpler than dedicated sentiment models
- Performance depends on the quality of input features
- Emoji handling is basic and could be improved
- Not optimized for specific domains (e.g., product reviews, social media)

## Future Improvements

- Domain-specific adapters
- Support for additional languages
- Expanded emoji and emoticon handling
- Enhanced negation context handling
- API service for online predictions 
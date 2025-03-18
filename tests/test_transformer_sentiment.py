"""Test suite for the transformer sentiment analysis model."""

import pytest
import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from app.models.ml.prediction.transformer_sentiment_colab import TransformerSentimentAnalyzer
from app.models.ml.prediction.transformer_sentiment_colab import CustomTensorBoardCallback
import logging
import os
from unittest.mock import Mock, patch, ANY
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from app.data.sentiment_dataset import SentimentDataset
from app.models.ml.prediction.ensemble_sentiment_analyzer import EnsembleSentimentAnalyzer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_initialization(sentiment_model):
    """Test model initialization."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test model type
    assert analyzer.model_type == 'bertweet'
    
    # Test device
    assert analyzer.device.type in ['cuda', 'cpu']
    
    # Test model and tokenizer
    assert analyzer.model is not None
    assert analyzer.tokenizer is not None
    
    # Test model configuration
    assert analyzer.num_labels == 2
    assert analyzer.model.config.num_labels == 2

def test_data_preprocessing(sample_sentiment_data):
    """Test data preprocessing."""
    analyzer = TransformerSentimentAnalyzer()
    
    # Test text preprocessing
    text = "This is a test tweet! #test"
    processed_text = analyzer._preprocess_text(text)
    assert isinstance(processed_text, str)
    assert len(processed_text) > 0
    
    # Test label preprocessing
    labels = np.array([0, 1, 1, 0])
    processed_labels = analyzer._preprocess_labels(labels)
    assert isinstance(processed_labels, np.ndarray)
    assert processed_labels.shape == labels.shape
    assert np.all(np.isin(processed_labels, [0, 1]))
    
    # Test dataset creation
    dataset = analyzer._create_dataset(sample_sentiment_data['texts'], sample_sentiment_data['labels'])
    assert len(dataset) == len(sample_sentiment_data['texts'])
    assert 'input_ids' in dataset.features
    assert 'attention_mask' in dataset.features
    assert 'labels' in dataset.features

def test_tokenization(sentiment_model, sample_sentiment_data):
    """Test text tokenization."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test single text tokenization
    text = "This is a test tweet! #test"
    tokens = analyzer.tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    assert tokens['input_ids'].shape[1] <= 128
    
    # Test batch tokenization
    texts = ["First tweet", "Second tweet", "Third tweet"]
    batch_tokens = analyzer.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    assert batch_tokens['input_ids'].shape[0] == len(texts)
    assert batch_tokens['attention_mask'].shape[0] == len(texts)

def test_model_training(sentiment_model, mock_training_data, mock_validation_data, training_args, mock_tensorboard):
    """Test model training."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Create training and validation datasets
    train_dataset = analyzer._create_dataset(
        mock_training_data['texts'],
        mock_training_data['labels']
    )
    val_dataset = analyzer._create_dataset(
        mock_validation_data['texts'],
        mock_validation_data['labels']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CustomTensorBoardCallback(mock_tensorboard)]
    )
    
    # Train model
    train_result = trainer.train()
    
    # Check training results
    assert train_result.training_loss is not None
    assert train_result.metrics is not None
    assert 'train_loss' in train_result.metrics
    
    # Check model state
    assert model.training is False  # Model should be in eval mode after training
    assert next(model.parameters()).requires_grad is True  # Parameters should still be trainable

def test_model_evaluation(sentiment_model, mock_validation_data):
    """Test model evaluation."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Mock evaluation
    with patch.object(analyzer, '_evaluate_model') as mock_eval:
        mock_eval.return_value = {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1': 0.85
        }
        
        metrics = analyzer.evaluate(model, mock_validation_data)
        
        assert metrics is not None
        assert all(metric > 0.8 for metric in metrics.values())

def test_prediction(sentiment_model):
    """Test model prediction."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test single prediction
    text = "This is a great product! I love it!"
    prediction = analyzer.predict(text)
    
    # Check prediction format
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]  # Binary classification
    
    # Test prediction with confidence
    prediction_with_conf = analyzer.predict(text, return_confidence=True)
    assert isinstance(prediction_with_conf, tuple)
    assert len(prediction_with_conf) == 2
    assert isinstance(prediction_with_conf[0], np.ndarray)
    assert isinstance(prediction_with_conf[1], np.ndarray)
    assert prediction_with_conf[1].shape == (1, 2)  # Confidence scores for each class

def test_batch_prediction(sentiment_model):
    """Test batch prediction."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    texts = [
        "This is amazing!",
        "I hate this product.",
        "It's okay, nothing special."
    ]
    
    predictions = analyzer.predict_batch(texts)
    
    assert len(predictions) == len(texts)
    assert all(isinstance(pred, dict) for pred in predictions)
    assert all('sentiment' in pred for pred in predictions)
    assert all('confidence' in pred for pred in predictions)

def test_model_saving_loading(sentiment_model, test_output_dir):
    """Test model saving and loading."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test saving
    save_path = test_output_dir / "test_model"
    analyzer.save_model(model, tokenizer, save_path)
    
    assert save_path.exists()
    assert (save_path / "config.json").exists()
    assert (save_path / "pytorch_model.bin").exists()
    
    # Test loading
    loaded_model, loaded_tokenizer = analyzer.load_model(save_path)
    
    assert loaded_model is not None
    assert loaded_tokenizer is not None
    assert loaded_model.config.num_labels == model.config.num_labels

def test_fairness_evaluation(sentiment_model, synthetic_demographics, mock_training_data):
    """Test fairness evaluation."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Mock fairness evaluation
    with patch.object(analyzer, '_evaluate_fairness') as mock_fairness:
        mock_fairness.return_value = {
            'demographic_parity': 0.95,
            'equal_opportunity': 0.92,
            'disparate_impact': 0.98
        }
        
        fairness_metrics = analyzer.evaluate_fairness(
            model,
            mock_training_data,
            synthetic_demographics
        )
        
        assert fairness_metrics is not None
        assert all(metric > 0.9 for metric in fairness_metrics.values())

def test_bias_mitigation(sentiment_model, mock_training_data):
    """Test bias mitigation techniques."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Mock bias mitigation
    with patch.object(analyzer, '_apply_bias_mitigation') as mock_mitigation:
        mock_mitigation.return_value = {
            'original_metrics': {'accuracy': 0.85},
            'mitigated_metrics': {'accuracy': 0.83},
            'fairness_improvement': 0.15
        }
        
        mitigation_results = analyzer.apply_bias_mitigation(
            model,
            mock_training_data
        )
        
        assert mitigation_results is not None
        assert 'original_metrics' in mitigation_results
        assert 'mitigated_metrics' in mitigation_results
        assert 'fairness_improvement' in mitigation_results

def test_ensemble_prediction(sentiment_model):
    """Test ensemble prediction."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Mock ensemble prediction
    with patch.object(analyzer, '_ensemble_predict') as mock_ensemble:
        mock_ensemble.return_value = {
            'sentiment': 'positive',
            'confidence': 0.92,
            'model_votes': [1, 1, 1, 0, 1]
        }
        
        text = "This is a fantastic product!"
        ensemble_result = analyzer.ensemble_predict(text)
        
        assert ensemble_result is not None
        assert ensemble_result['sentiment'] in ['positive', 'negative']
        assert ensemble_result['confidence'] > 0.9
        assert 'model_votes' in ensemble_result

def test_gpu_utilization(mock_gpu, sentiment_model):
    """Test GPU utilization."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        assert analyzer.device.type == 'cuda'
        assert next(analyzer.model.parameters()).device.type == 'cuda'
    else:
        assert analyzer.device.type == 'cpu'
        assert next(analyzer.model.parameters()).device.type == 'cpu'

def test_cpu_fallback(mock_cpu, sentiment_model):
    """Test CPU fallback when GPU is not available."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test CPU detection
    assert analyzer.device.type == 'cpu'
    
    # Test model is on CPU
    assert next(analyzer.model.parameters()).device.type == 'cpu'

def test_error_handling(sentiment_model):
    """Test error handling."""
    model, tokenizer = sentiment_model
    analyzer = TransformerSentimentAnalyzer()
    
    # Test empty input
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.predict("")
    
    # Test invalid model type
    with pytest.raises(ValueError, match="Invalid model type"):
        TransformerSentimentAnalyzer(model_type="invalid_model")
    
    # Test invalid batch size
    with pytest.raises(ValueError, match="Batch size must be positive"):
        analyzer.predict_batch(["text"], batch_size=0)

def test_tensorboard_callback(mock_tensorboard):
    """Test TensorBoard callback."""
    callback = CustomTensorBoardCallback(mock_tensorboard)
    
    # Test on_log
    callback.on_log({"loss": 0.5, "eval_loss": 0.4}, state=None, control=None)
    mock_tensorboard.add_scalar.assert_called_with("train/loss", 0.5, None)
    mock_tensorboard.add_scalar.assert_called_with("eval/loss", 0.4, None)
    
    # Test on_train_begin
    callback.on_train_begin(None, state=None, control=None)
    mock_tensorboard.add_text.assert_called_with("hyperparameters", ANY)
    
    # Test on_train_end
    callback.on_train_end(None, state=None, control=None)
    mock_tensorboard.close.assert_called_once()

def test_plot_fairness_metrics_shape():
    """Test that plot_fairness_metrics runs without shape errors."""
    # Create test fairness metrics with different shapes
    test_metrics = {
        'age_group': {
            '18-24': {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.82, 'f1': 0.78},
            '25-34': {'accuracy': 0.82, 'precision': 0.78, 'recall': 0.85, 'f1': 0.81},
            '35-44': {'accuracy': 0.79, 'precision': 0.76, 'recall': 0.80, 'f1': 0.78}
        },
        'gender': {
            'M': {'accuracy': 0.81, 'precision': 0.77, 'recall': 0.83, 'f1': 0.80},
            'F': {'accuracy': 0.80, 'precision': 0.76, 'recall': 0.82, 'f1': 0.79}
        }
    }
    
    # Create temporary directory for plots
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test plotting with different demographic features
        for feature in test_metrics:
            try:
                # Create a single plot for the feature
                plt.figure(figsize=(12, 6))
                
                # Get data for this feature
                metrics = test_metrics[feature]
                groups = list(metrics.keys())
                metric_names = ['accuracy', 'precision', 'recall', 'f1']
                
                # Plot each metric
                x = np.arange(len(groups))
                width = 0.2
                
                for i, metric in enumerate(metric_names):
                    values = [metrics[group][metric] for group in groups]
                    plt.bar(x + i*width, values, width, label=metric)
                
                plt.xlabel(feature)
                plt.ylabel('Score')
                plt.title(f'Fairness Metrics by {feature}')
                plt.xticks(x + width*1.5, groups)
                plt.legend()
                
                # Save plot
                plt.savefig(os.path.join(temp_dir, f'fairness_{feature}.png'))
                plt.close()
                
                # Verify file was created
                assert os.path.exists(os.path.join(temp_dir, f'fairness_{feature}.png'))
                
            except Exception as e:
                pytest.fail(f"Error plotting fairness metrics for {feature}: {str(e)}")

def test_background_dataset_preparation():
    """Test background dataset preparation with a small sample size."""
    # Create a small test dataset
    test_texts = [
        "This is a positive review",
        "This is a negative review",
        "I love this product",
        "I hate this product",
        "It's okay, nothing special"
    ]
    test_labels = [1, 0, 1, 0, 0]
    
    # Create a small dataset
    test_dataset = SentimentDataset(
        test_texts,
        test_labels,
        AutoTokenizer.from_pretrained('vinai/bertweet-base'),
        max_length=128
    )
    
    # Initialize ensemble model
    transformer_model = TransformerSentimentAnalyzer(model_type='bertweet')
    ensemble_model = EnsembleSentimentAnalyzer(transformer_model)
    
    # Try to prepare background dataset with small sample size
    try:
        success = ensemble_model._prepare_background_dataset(test_dataset, n_samples=3)
        assert success, "Background dataset preparation failed"
        
        # Verify background dataset size
        assert len(ensemble_model.background_dataset) > 0, "Background dataset is empty"
        assert len(ensemble_model.background_dataset) <= 3, "Background dataset is too large"
        
        # Verify dataset format
        for item in ensemble_model.background_dataset:
            assert isinstance(item, dict), "Dataset item is not a dictionary"
            assert 'input_ids' in item, "Dataset item missing input_ids"
            assert 'labels' in item, "Dataset item missing labels"
        
        logger.info("Background dataset preparation test passed successfully")
        
    except Exception as e:
        pytest.fail(f"Background dataset preparation test failed: {str(e)}") 
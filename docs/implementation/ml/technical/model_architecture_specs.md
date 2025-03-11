# Model Architecture Specifications

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document provides detailed technical specifications for the machine learning model architectures used in the WITHIN platform. It covers the structural design, component interactions, and implementation details for all core prediction models.

## Table of Contents

1. [Ad Score Predictor Architecture](#ad-score-predictor-architecture)
2. [Account Health Predictor Architecture](#account-health-predictor-architecture)
3. [Ad Sentiment Analyzer Architecture](#ad-sentiment-analyzer-architecture)
4. [Shared Components](#shared-components)
5. [Implementation Guidelines](#implementation-guidelines)
6. [Performance Requirements](#performance-requirements)
7. [Validation Protocols](#validation-protocols)
8. [Advanced Architectural Components](#advanced-architectural-components)

## Ad Score Predictor Architecture

### Core Architecture

The Ad Score Predictor uses a hybrid architecture combining gradient boosting and deep learning components:

#### Ensemble Structure

```
┌─────────────────────────────┐
│         Meta-Ensemble       │
└───────────────┬─────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼───┐             ┌─────▼───┐
│ GBDT  │             │  Deep   │
│ Model │             │ Learning│
└───────┘             └─────────┘
```

#### Component Details

1. **Gradient Boosting Decision Tree (GBDT)**
   - Implementation: XGBoost
   - Trees: 150
   - Max Depth: 8
   - Learning Rate: 0.01
   - L1 Regularization: 0.5
   - L2 Regularization: 1.0
   - Feature Fraction: 0.8
   - Bagging Fraction: 0.7

2. **Deep Learning Component**
   - Framework: PyTorch
   - Structure: Multi-modal network with attention mechanisms
   - Text Encoder: Transformer-based (6 layers, 8 attention heads)
   - Visual Encoder: Modified EfficientNet-B2
   - Fusion Layer: Cross-modal attention
   - Hidden Layers: [512, 256, 128]
   - Activation: GELU
   - Dropout: 0.2
   - Batch Normalization: Applied after each hidden layer

3. **Meta-Ensemble**
   - Stacking approach with calibration
   - Weighting: Learned through Bayesian optimization
   - Initial Weights: [0.65, 0.35] for GBDT and DL respectively
   - Calibration: Isotonic regression

### Feature Processing

The architecture includes specialized feature processing components:

1. **Text Processing Pipeline**
   ```python
   text_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='constant', fill_value='')),
       ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
       ('truncated_svd', TruncatedSVD(n_components=100))
   ])
   ```

2. **Numerical Feature Pipeline**
   ```python
   numerical_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='median')),
       ('scaler', StandardScaler())
   ])
   ```

3. **Categorical Feature Pipeline**
   ```python
   categorical_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='most_frequent')),
       ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
   ])
   ```

4. **Visual Feature Pipeline**
   ```python
   visual_pipeline = Pipeline([
       ('resize', ImageResizer(224, 224)),
       ('normalize', ImageNormalizer()),
       ('feature_extractor', EfficientNetFeatureExtractor(model='b2'))
   ])
   ```

### Cross-Modal Attention Implementation

The Cross-Modal Attention mechanism is implemented as follows:

```python
class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for text and image features"""
    def __init__(self, visual_dim=512, text_dim=100, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features, text_features):
        # Project features to common space
        visual_proj = self.norm1(self.visual_proj(visual_features))
        text_proj = self.norm2(self.text_proj(text_features))
        
        # Compute attention scores
        attention_scores = torch.matmul(visual_proj, text_proj.transpose(-2, -1))
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context_features = torch.matmul(attention_probs, text_proj)
        
        # Concatenate original and context features
        enhanced_visual = torch.cat([visual_proj, context_features], dim=-1)
        return enhanced_visual
```

## Account Health Predictor Architecture

### Core Architecture

The Account Health Predictor uses a time-series-focused architecture:

#### Component Structure

```
┌──────────────────────┐
│ Time Series Pipeline │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LSTM Sequence Model │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────┐
│  Anomaly Detector    │◀───│ Historical Data  │
└──────────┬───────────┘    └──────────────────┘
           │
           ▼
┌──────────────────────┐
│  XGBoost Classifier  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Health Score Calibrator │
└──────────────────────┘
```

#### Component Details

1. **Time Series Pipeline**
   - Feature Extraction: Statistical features from time series
   - Sequence Length: 30-90 days
   - Aggregation Levels: Daily, Weekly, Monthly
   - Transformation: Log, Difference, Moving Average
   - Missing Value Strategy: Linear interpolation

2. **LSTM Sequence Model**
   - Cell Type: LSTM with peephole connections
   - Hidden Size: 128
   - Layers: 2
   - Bidirectional: Yes
   - Dropout: 0.3
   - Recurrent Dropout: 0.2
   - Attention: Temporal self-attention

3. **Anomaly Detector**
   - Primary Algorithm: Isolation Forest
   - Supporting Algorithm: DBSCAN
   - Contamination: Auto-determined from data
   - Feature Subset: Platform-specific key metrics
   - Ensemble Method: Majority voting

4. **XGBoost Classifier**
   - Trees: 100
   - Max Depth: 6
   - Learning Rate: 0.05
   - Objective: Multi:softprob (for health categories)
   - Evaluation Metric: AUC
   - Early Stopping: 10 rounds

### Implementation Notes

The Account Health Predictor uses a modular design with these key components:

```python
class TimeSeriesFeatureExtractor:
    def __init__(self, sequence_length=30, aggregation='daily'):
        self.sequence_length = sequence_length
        self.aggregation = aggregation
    
    def transform(self, data):
        # Extract statistical features from time series
        # Implementation details in account_health_predictor.py
        pass

class LSTMSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3
        )
        self.attention = TemporalSelfAttention(
            hidden_size * (2 if bidirectional else 1)
        )
    
    def forward(self, x):
        # LSTM sequence processing with attention
        # Implementation details in account_health_predictor.py
        pass
```

## Ad Sentiment Analyzer Architecture

### Core Architecture

The Ad Sentiment Analyzer uses a transformer-based architecture:

#### Component Structure

```
┌───────────────────┐
│  Text Preprocessing │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  BERT Encoder     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Sentiment Head   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Calibration      │
└───────────────────┘
```

#### Component Details

1. **Text Preprocessing**
   - Tokenizer: BertTokenizer
   - Max Length: 512 tokens
   - Special Tokens: [CLS], [SEP]
   - Truncation: Yes
   - Padding: To max length
   - Handling Emojis: Convert to text

2. **BERT Encoder**
   - Base Model: BERT-base (or DistilBERT)
   - Layers Used: Last 4 layers
   - Pooling: Attention-weighted
   - Fine-tuning: Last 3 layers only

3. **Sentiment Head**
   - Architecture: MLP
   - Hidden Layers: [768, 256, 64]
   - Output: 3 classes (positive, neutral, negative)
   - Activation: ReLU for hidden, Softmax for output
   - Dropout: 0.1

4. **Calibration**
   - Method: Temperature scaling
   - Validation: Brier score

### Implementation Notes

The key implementation aspects include:

```python
class SentimentAnalyzer(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Process through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get CLS token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Classify
        logits = self.classifier(pooled_output)
        return logits
```

## Shared Components

Several components are shared across multiple models:

### Feature Store

```python
class FeatureStore:
    def __init__(self, cache_ttl=3600):
        self.cache_ttl = cache_ttl
        self._cache = {}
        
    def get_feature(self, feature_name, entity_id):
        # Retrieve feature from store or compute
        # Implementation details in feature_store.py
        pass
        
    def put_feature(self, feature_name, entity_id, value):
        # Store feature value
        # Implementation details in feature_store.py
        pass
```

### Model Registry

```python
class ModelRegistry:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        
    def register_model(self, model_name, model_version, model_path, metadata):
        # Register model in registry
        # Implementation details in model_registry.py
        pass
        
    def load_model(self, model_name, model_version=None):
        # Load model from registry
        # Implementation details in model_registry.py
        pass
```

### Explanation Generator

```python
class ShapExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model, shap.sample(X_train, 100))
        
    def explain(self, x):
        # Generate SHAP explanations
        # Implementation details in explainer.py
        pass
```

## Implementation Guidelines

When implementing these architectures, follow these guidelines:

1. **Modularity**: Each component should be implemented as a separate class
2. **Type Safety**: Use Python type hints throughout
3. **Serialization**: All models must implement `save` and `load` methods
4. **Documentation**: All classes must have Google-style docstrings
5. **Testing**: Unit tests required for each component
6. **Error Handling**: Implement appropriate exception handling
7. **Logging**: Use standard logging framework
8. **Performance**: Optimize critical paths
9. **Reproducibility**: Set random seeds for all stochastic components

## Performance Requirements

Model implementations must meet these performance requirements:

1. **Inference Time**:
   - Ad Score Predictor: < 200ms per prediction
   - Account Health Predictor: < 500ms per account
   - Ad Sentiment Analyzer: < 100ms per text

2. **Memory Usage**:
   - Ad Score Predictor: < 1.5GB
   - Account Health Predictor: < 1GB
   - Ad Sentiment Analyzer: < 2GB

3. **Throughput**:
   - Ad Score Predictor: > 50 predictions/second
   - Account Health Predictor: > 20 accounts/second
   - Ad Sentiment Analyzer: > 100 texts/second

4. **Batch Processing**:
   - Ad Score Predictor: Scale linearly up to 100 items
   - Account Health Predictor: Scale linearly up to 50 items
   - Ad Sentiment Analyzer: Scale linearly up to 200 items

## Validation Protocols

Implementation validation requires:

1. **Functional Tests**: Verify correct outputs for valid inputs
2. **Performance Tests**: Measure inference time and memory usage
3. **Regression Tests**: Compare against previous versions
4. **Integration Tests**: Validate with real data flows
5. **Error Handling Tests**: Verify graceful handling of bad inputs
6. **Reproducibility Tests**: Verify consistent outputs with fixed seeds

## Advanced Architectural Components

This section documents specialized architectural components used in the WITHIN ML models that require detailed technical explanation due to their complexity and importance in the system.

### Cross-Modal Attention Mechanism

The `CrossModalAttention` component is a critical element in our multi-modal models, enabling the integration of text and visual features through an attention mechanism. This component is primarily used in the Ad Score Predictor to create rich representations that combine information from both textual and visual elements of advertisements.

#### Architecture Overview

```
                ┌──────────────────┐
                │    Visual Input  │
                └──────────┬───────┘
                           ▼
                ┌──────────────────┐
                │   Visual Proj    │
                └──────────┬───────┘
                           │         ┌──────────────┐
                           │         │  Text Input  │
                           │         └──────┬───────┘
                           │                ▼
                           │         ┌──────────────┐
                           │         │  Text Proj   │
                           │         └──────┬───────┘
                           ▼                ▼
                    ┌─────────────────────────────┐
                    │      Multi-Head Attention   │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │     Layer Normalization     │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │        Output Projection    │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │      Fused Representation   │
                    └─────────────────────────────┘
```

#### Implementation Details

The `CrossModalAttention` module is implemented as a PyTorch `nn.Module` with the following components:

```python
class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for text and image features"""
    def __init__(self, visual_dim=512, text_dim=100, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention
        self.n_heads = 4
        self.head_dim = hidden_dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
```

The forward pass performs these operations:
1. Project visual and text features to a common hidden dimension
2. Apply layer normalization
3. Compute multi-head attention between the modalities
4. Apply output projection to create the fused representation

#### Key Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| visual_dim | 512 | Dimensionality of visual feature input |
| text_dim | 100 | Dimensionality of text feature input |
| hidden_dim | 256 | Dimensionality of internal representations |
| n_heads | 4 | Number of attention heads |

#### Usage Example

The CrossModalAttention module is used in the Ad Score Predictor as follows:

```python
# Extract features from modalities
text_features = self.text_encoder(ad_text)
visual_features = self.visual_encoder(ad_image)

# Apply cross-modal attention
fused_features = self.cross_modal_attention(
    visual_features=visual_features,
    text_features=text_features
)

# Use fused features for prediction
ad_score = self.prediction_head(fused_features)
```

#### Performance Characteristics

- **Computational Complexity**: O(n²d) where n is sequence length and d is hidden dimension
- **Memory Usage**: Proportional to visual_dim + text_dim + 4*hidden_dim
- **Inference Time**: ~5-8ms on V100 GPU for typical input sizes
- **Hardware Acceleration**: Optimized for GPU execution with CUDA

#### Implementation Considerations

When working with the CrossModalAttention component:

1. **Dimensionality Balance**: Ensure visual_dim and text_dim are appropriately balanced for your specific modalities
2. **Head Count**: The number of attention heads should divide evenly into hidden_dim
3. **Normalization**: Layer normalization is critical for stable training
4. **Gradient Flow**: Residual connections help with gradient flow in deeper networks
5. **Numerical Stability**: The scaling factor (head_dim ** -0.5) prevents softmax saturation

This component is central to the multi-modal capabilities of our ad scoring system, enabling rich interactions between textual and visual content that significantly improve prediction accuracy.

---

**Document Revision History**:
- v1.0 (2023-01-15): Initial specification
- v1.1 (2023-03-22): Updated performance requirements
- v1.2 (2023-06-10): Added cross-modal attention details
- v2.0 (2023-09-05): Major update with component diagrams 
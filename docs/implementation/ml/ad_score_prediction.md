# Ad Score Prediction Implementation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

The WITHIN Ad Score Predictor is a machine learning system that evaluates the potential performance of digital advertisements before they are published. This document details the technical implementation of the prediction system, including the model architecture, training methodology, and deployment approach.

## Table of Contents

1. [Architectural Design](#architectural-design)
2. [System Components](#system-components)
3. [Model Architecture](#model-architecture)
4. [Prediction Pipeline](#prediction-pipeline)
5. [Implementation Details](#implementation-details)
6. [Performance Optimization](#performance-optimization)
7. [Integration Points](#integration-points)
8. [Example Usage](#example-usage)
9. [Troubleshooting](#troubleshooting)
10. [Future Development](#future-development)

## Architectural Design

The Ad Score Prediction system follows a layered architecture pattern with API, service, feature engineering, prediction, explanation, and response formatting layers. 

## System Components

### 1. Ad Score Predictor

The core component is defined in `app/models/ml/prediction/ad_score_predictor.py` and implements:

- Model loading and initialization
- Prediction request handling
- Feature transformation
- Score calibration
- Explanation generation

### 2. Feature Engineering

The feature engineering pipeline (detailed in [Feature Engineering Documentation](/docs/implementation/ml/feature_engineering.md)) handles:

- Text feature extraction
- Campaign feature processing
- Temporal feature generation
- Cross-platform feature synthesis
- Feature normalization

### 3. Model Registry

The `app/models/ml/registry.py` provides:

- Model versioning
- Model lifecycle management
- A/B testing capabilities
- Fallback model support

### 4. Explanation Engine

The explanation system in `app/models/ml/explainers` provides:

- SHAP-based feature attribution
- Contrastive explanations
- Recommendations for improvement 

## Model Architecture

The Ad Score Predictor uses an ensemble architecture combining:

1. **Primary Model**: XGBoost gradient boosting classifier (for categorical scores) or regressor (for continuous scores)
2. **Text Embedding Model**: DistilBERT fine-tuned on advertising content
3. **Time Series Component**: LSTM network for temporal pattern recognition
4. **Calibration Layer**: Isotonic regression for probability calibration

### Implementation Example

```python
class AdScorePredictor:
    """Ad Score prediction model with XGBoost ensemble and neural components."""
    
    def __init__(
        self, 
        model_path: str,
        embedding_model_path: str,
        calibration_model_path: str,
        feature_config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the Ad Score Predictor with all required components."""
        self.device = device
        self.feature_config = feature_config
        
        # Load models
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(model_path)
        
        self.embedding_model = AutoModel.from_pretrained(embedding_model_path)
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        with open(calibration_model_path, "rb") as f:
            self.calibration_model = pickle.load(f)
            
        # Initialize processors
        self.text_processor = TextFeatureProcessor(
            self.embedding_model, 
            self.feature_config["text_features"]
        )
        self.campaign_processor = CampaignFeatureProcessor(
            self.feature_config["campaign_features"]
        )
        self.temporal_processor = TemporalFeatureProcessor(
            self.feature_config["temporal_features"]
        )
        
        # Initialize explainer
        self.explainer = AdScoreExplainer(self.xgb_model, feature_config) 
```

## Prediction Pipeline

The prediction process follows these steps:

1. **Request Validation**: Input validation using Pydantic schemas
2. **Feature Extraction**: Text, campaign, and historical data processed
3. **Model Inference**: Features passed through the ensemble model
4. **Score Calibration**: Raw scores calibrated to final performance range
5. **Confidence Calculation**: Uncertainty estimation based on feature distribution
6. **Explanation Generation**: SHAP values calculated for feature contributions
7. **Response Formatting**: Results structured into API response

## Implementation Details

### Key Classes and Modules

1. **AdScorePredictor**: Main prediction class integrating all components
2. **TextFeatureProcessor**: Handles text embedding and NLP feature extraction
3. **CampaignFeatureProcessor**: Processes campaign settings and metadata
4. **TemporalFeatureProcessor**: Extracts patterns from historical data
5. **AdScoreExplainer**: Generates explanations for predictions
6. **ScoreCalibrator**: Calibrates raw model outputs to final scores

### Technology Stack

- **Python 3.9+**: Core implementation language
- **PyTorch**: For neural network components and text embedding
- **XGBoost**: For gradient boosting models
- **scikit-learn**: For feature processing and calibration
- **FastAPI**: For API endpoints
- **SHAP**: For model explanations
- **Pydantic**: For data validation

### Code Structure

```
app/models/ml/prediction/
├── __init__.py
├── ad_score_predictor.py        # Main prediction class
├── processors/
│   ├── __init__.py
│   ├── text_processor.py        # Text feature extraction
│   ├── campaign_processor.py    # Campaign feature processing
│   └── temporal_processor.py    # Historical data processing
├── explainers/
│   ├── __init__.py
│   ├── shap_explainer.py        # SHAP-based explanation engine
│   └── suggestion_generator.py  # Improvement suggestion logic
└── calibration/
    ├── __init__.py
    └── score_calibrator.py      # Score calibration models
```

## Performance Optimization

## Integration Points

### API Integration

The model is exposed through a FastAPI endpoint:

```python
@router.post("/predict/ad-score", response_model=AdScorePredictionResponse)
async def predict_ad_score(
    request: AdScorePredictionRequest,
    predictor: AdScorePredictor = Depends(get_ad_score_predictor)
):
    """Predict performance score for an advertisement."""
    try:
        # Extract request data
        ad_text = request.ad_text
        campaign_data = request.campaign_data
        historical_data = request.historical_data
        
        # Make prediction
        result = predictor.predict(
            ad_text=ad_text,
            campaign_data=campaign_data,
            historical_data=historical_data,
            return_explanations=request.include_explanations
        )
        
        return AdScorePredictionResponse(
            score=result["score"],
            confidence=result["confidence"],
            score_category=result["score_category"],
            explanations=result.get("explanations"),
            improvement_suggestions=result.get("improvement_suggestions")
        )
    except Exception as e:
        # Log error
        logger.error(f"Ad score prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
```

### Python SDK Integration

The model is accessible through the Python SDK:

```python
from within.client import WITHINClient

# Initialize client
client = WITHINClient(api_key="your_api_key")

# Predict ad score
prediction = client.predict_ad_score(
    ad_text="Experience our revolutionary new product today!",
    campaign_data={
        "platform": "facebook",
        "objective": "conversions",
        "audience_size": 1000000,
        "budget": 5000,
        "industry": "retail"
    },
    include_explanations=True
)

# Access results
print(f"Ad Score: {prediction.score}")
print(f"Confidence: {prediction.confidence}")
print(f"Category: {prediction.score_category}")
```

## Example Usage

### Basic Prediction

```python
from app.models.ml.prediction import AdScorePredictor

# Initialize predictor
predictor = AdScorePredictor(
    model_path="models/ad_score/xgboost_model.json",
    embedding_model_path="models/ad_score/text_embedding",
    calibration_model_path="models/ad_score/calibration.pkl",
    feature_config=config.FEATURE_CONFIG
)

# Make prediction
result = predictor.predict(
    ad_text="Limited time offer! Shop now and save 50% on all products.",
    campaign_data={
        "platform": "instagram",
        "objective": "conversions",
        "audience_size": 500000,
        "budget": 1000,
        "industry": "fashion"
    }
)

print(f"Ad Score: {result['score']}")
print(f"Confidence: {result['confidence']}")
print(f"Category: {result['score_category']}")
```

## Troubleshooting

### Common Issues

#### 1. Low Confidence Scores

If predictions consistently have low confidence scores:

```python
def diagnose_low_confidence(predictor, ad_text, campaign_data):
    """Diagnose why a prediction has low confidence."""
    # Extract features without prediction
    text_features = predictor.text_processor.process(ad_text)
    campaign_features = predictor.campaign_processor.process(campaign_data)
    
    # Check for out-of-distribution features
    text_novelty = predictor._calculate_text_novelty(text_features)
    campaign_novelty = predictor._calculate_campaign_novelty(campaign_features)
    
    print(f"Text novelty score: {text_novelty:.4f}")
    print(f"Campaign novelty score: {campaign_novelty:.4f}")
    
    # Provide guidance
    if text_novelty > 0.8:
        print("The ad text contains unusual patterns not well represented in training data")
    if campaign_novelty > 0.8:
        print("The campaign settings are unusual compared to the training data")
```

#### 2. Inconsistent Predictions

If predictions are inconsistent between calls:

```python
def check_prediction_stability(predictor, ad_text, campaign_data, n_trials=10):
    """Check stability of predictions across multiple calls."""
    scores = []
    for _ in range(n_trials):
        result = predictor.predict(ad_text, campaign_data)
        scores.append(result["score"])
    
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    
    print(f"Mean score: {mean_score:.4f}")
    print(f"Score variance: {variance:.6f}")
    
    if variance > 0.01:
        print("WARNING: High prediction variance detected")
        print("This may indicate a problem with model stability")
```

## Future Development

Planned enhancements to the Ad Score Prediction system include:

1. **Multi-modal Support**: Adding image and video content analysis
2. **Automated A/B Testing**: Integration with experimental design
3. **Personalized Scoring**: Audience-specific performance prediction
4. **Continuous Learning**: Online model updates from feedback
5. **Multi-platform Optimization**: Specialized models for each ad platform

### Development Roadmap

```
Q3 2023: Add multi-modal support for image analysis
Q4 2023: Implement automated A/B testing framework
Q1 2024: Develop personalized scoring capabilities
Q2 2024: Deploy continuous learning pipeline
Q3 2024: Launch platform-specific models
```

## Related Documentation

For more information, refer to the following documentation:

- [Feature Engineering Documentation](/docs/implementation/ml/feature_engineering.md) *(Implemented)*
- [Model Training Process](/docs/implementation/ml/model_training.md) *(Implemented)*
- [Model Evaluation](/docs/implementation/ml/model_evaluation.md) *(Planned - Not yet implemented)*
- [Ad Score Model Card](/docs/implementation/ml/model_card_ad_score_predictor.md) *(Implemented)*
- [API Documentation](/docs/api/overview.md) *(Implemented)*

> **Note**: Some of the linked documents above are currently planned but not yet implemented. Please refer to the [Documentation Tracker](/docs/implementation/documentation_tracker.md) for the current status of all documentation.

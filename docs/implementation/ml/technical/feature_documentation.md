# Feature Documentation

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document provides comprehensive documentation for all features used in the WITHIN ML models. It covers feature definitions, extraction methods, preprocessing steps, and usage guidelines to ensure consistent implementation across the ML system.

## Table of Contents

1. [Feature Categories](#feature-categories)
2. [Feature Extraction Process](#feature-extraction-process)
3. [Feature Definitions](#feature-definitions)
4. [Preprocessing Pipelines](#preprocessing-pipelines)
5. [Feature Selection](#feature-selection)
6. [Feature Engineering Guidelines](#feature-engineering-guidelines)
7. [Feature Store Integration](#feature-store-integration)
8. [Feature Monitoring](#feature-monitoring)
9. [Versioning and Compatibility](#versioning-and-compatibility)
10. [Feature Development Process](#feature-development-process)

## Feature Categories

Features in the WITHIN platform are organized into these main categories:

### Ad Content Features

Features derived from the content of advertisements:

| Category | Description | Examples |
|----------|-------------|----------|
| Text Features | Derived from ad copy | Word embeddings, sentiment scores, topic distributions |
| Visual Features | Extracted from ad images | Color schemes, object detection, composition metrics |
| Video Features | Extracted from video ads | Scene transitions, pacing, audio features |
| Structural Features | Based on ad format | Layout scores, element positioning, whitespace utilization |
| CTA Features | Related to calls-to-action | CTA clarity, position, action verbs used |

### Performance Features

Features based on historical ad performance:

| Category | Description | Examples |
|----------|-------------|----------|
| Engagement Metrics | User interaction metrics | CTR, engagement rate, view duration |
| Conversion Metrics | Conversion-related | Conversion rate, CPA, ROAS |
| Temporal Patterns | Time-based patterns | Day/hour performance, seasonality indices |
| Comparative Metrics | Benchmarking metrics | Performance vs. industry average, relative improvement |
| Audience Response | Audience-specific metrics | Demographic response rates, audience affinity scores |

### Account Features

Features related to the overall advertising account:

| Category | Description | Examples |
|----------|-------------|----------|
| Account Structure | Organization metrics | Campaign count, ad group diversity, targeting breadth |
| Budget Allocation | Spending patterns | Budget distribution, spend pacing, allocation efficiency |
| Historical Performance | Overall account trends | Account growth trajectory, performance stability |
| Platform Utilization | Platform-specific metrics | Feature adoption rate, optimization score |
| Account Health | Overall health indicators | Policy compliance rate, quality score averages |

### Contextual Features

Features capturing contextual information:

| Category | Description | Examples |
|----------|-------------|----------|
| Industry Context | Industry-specific data | Vertical benchmarks, industry maturity |
| Platform Context | Platform-specific data | Platform-specific performance benchmarks |
| Seasonal Context | Temporal context | Seasonal adjustment factors, holiday relevance |
| Competitive Context | Market positioning | Share of voice, competitive density |
| External Factors | External influences | Market trends, economic indicators |

## Feature Extraction Process

### Process Overview

The feature extraction pipeline consists of these stages:

1. **Data Collection**: Gather raw data from various sources
2. **Raw Feature Extraction**: Extract basic features from raw data
3. **Feature Transformation**: Apply transformations to raw features
4. **Feature Combination**: Create composite features
5. **Feature Selection**: Select relevant features for models
6. **Feature Validation**: Validate feature quality and distributions
7. **Feature Persistence**: Store computed features in feature store

### Implementation

The feature extraction is implemented in `app/models/ml/features/extraction.py`:

```python
from typing import Dict, Any, List, Optional
import pandas as pd
from within.models.ml.features.extractors import (
    TextFeatureExtractor,
    VisualFeatureExtractor,
    PerformanceFeatureExtractor,
    AccountFeatureExtractor,
    ContextualFeatureExtractor
)

class FeatureExtractionPipeline:
    """Pipeline for extracting features from raw data"""
    
    def __init__(self):
        self.extractors = {
            "text": TextFeatureExtractor(),
            "visual": VisualFeatureExtractor(),
            "performance": PerformanceFeatureExtractor(),
            "account": AccountFeatureExtractor(),
            "contextual": ContextualFeatureExtractor()
        }
    
    def extract_features(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """Extract all features from raw data
        
        Args:
            raw_data: Dictionary containing raw data for feature extraction
            
        Returns:
            DataFrame containing all extracted features
        """
        features = {}
        
        # Extract features from each extractor
        for name, extractor in self.extractors.items():
            if name in raw_data:
                features.update(extractor.extract(raw_data[name]))
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([features])
        
        return feature_df
```

## Feature Definitions

### Text Features

#### Word Embedding Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `text_embedding_headline` | BERT embedding of headline | array[768] | (-∞, +∞) | Headline text |
| `text_embedding_description` | BERT embedding of description | array[768] | (-∞, +∞) | Description text |
| `text_embedding_cta` | BERT embedding of CTA | array[768] | (-∞, +∞) | CTA text |
| `text_embedding_combined` | Combined text embedding | array[768] | (-∞, +∞) | All text fields |

#### Semantic Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `sentiment_score` | Overall sentiment score | float | [-1.0, 1.0] | All text |
| `emotion_joy` | Joy emotion score | float | [0.0, 1.0] | All text |
| `emotion_trust` | Trust emotion score | float | [0.0, 1.0] | All text |
| `clarity_score` | Message clarity score | float | [0.0, 100.0] | All text |
| `persuasiveness_score` | Persuasiveness score | float | [0.0, 100.0] | All text |
| `topic_distribution` | Topic distribution vector | array[20] | [0.0, 1.0] | All text |

#### Linguistic Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `reading_ease` | Flesch reading ease score | float | [0.0, 100.0] | All text |
| `word_count` | Total word count | int | [0, ∞) | All text |
| `question_count` | Number of questions | int | [0, ∞) | All text |
| `exclamation_count` | Number of exclamations | int | [0, ∞) | All text |
| `uppercase_ratio` | Ratio of uppercase characters | float | [0.0, 1.0] | All text |
| `action_verb_count` | Number of action verbs | int | [0, ∞) | All text |

### Visual Features

#### Color Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `color_brightness` | Average brightness | float | [0.0, 1.0] | Image |
| `color_contrast` | Color contrast | float | [0.0, 1.0] | Image |
| `color_saturation` | Color saturation | float | [0.0, 1.0] | Image |
| `dominant_colors` | Dominant color palette | array[5, 3] | [0, 255] | Image |
| `color_harmony` | Color harmony score | float | [0.0, 1.0] | Image |

#### Composition Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `visual_complexity` | Visual complexity score | float | [0.0, 1.0] | Image |
| `edge_density` | Edge density | float | [0.0, 1.0] | Image |
| `symmetry_score` | Symmetry score | float | [0.0, 1.0] | Image |
| `focal_point_strength` | Focal point strength | float | [0.0, 1.0] | Image |
| `rule_of_thirds_score` | Rule of thirds adherence | float | [0.0, 1.0] | Image |

#### Content Features

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `object_count` | Number of detected objects | int | [0, ∞) | Image |
| `face_count` | Number of faces detected | int | [0, ∞) | Image |
| `face_emotion_vector` | Face emotion vector | array[7] | [0.0, 1.0] | Image |
| `text_overlay_ratio` | Ratio of text overlay | float | [0.0, 1.0] | Image |
| `product_presence` | Product presence score | float | [0.0, 1.0] | Image |

### Performance Features

#### Engagement Metrics

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `ctr_historical` | Historical click-through rate | float | [0.0, 1.0] | Performance data |
| `engagement_rate` | Engagement rate | float | [0.0, 1.0] | Performance data |
| `video_completion_rate` | Video completion rate | float | [0.0, 1.0] | Performance data |
| `avg_session_duration` | Average session duration | float | [0.0, ∞) | Performance data |
| `bounce_rate` | Bounce rate | float | [0.0, 1.0] | Performance data |

#### Conversion Metrics

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `conversion_rate` | Conversion rate | float | [0.0, 1.0] | Performance data |
| `cost_per_acquisition` | Cost per acquisition | float | [0.0, ∞) | Performance data |
| `return_on_ad_spend` | Return on ad spend | float | [0.0, ∞) | Performance data |
| `average_order_value` | Average order value | float | [0.0, ∞) | Performance data |
| `customer_lifetime_value` | Customer lifetime value | float | [0.0, ∞) | Performance data |

### Account Features

#### Structure Metrics

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `campaign_count` | Number of campaigns | int | [0, ∞) | Account data |
| `ad_group_diversity` | Ad group diversity score | float | [0.0, 1.0] | Account data |
| `targeting_breadth` | Targeting breadth score | float | [0.0, 1.0] | Account data |
| `asset_diversity` | Asset diversity score | float | [0.0, 1.0] | Account data |
| `account_maturity` | Account maturity score | float | [0.0, 1.0] | Account data |

#### Health Metrics

| Feature | Description | Type | Range | Source |
|---------|-------------|------|-------|--------|
| `quality_score_avg` | Average quality score | float | [1.0, 10.0] | Account data |
| `policy_compliance_rate` | Policy compliance rate | float | [0.0, 1.0] | Account data |
| `account_stability` | Account stability score | float | [0.0, 1.0] | Account data |
| `optimization_score` | Platform optimization score | float | [0.0, 100.0] | Account data |
| `feature_adoption_rate` | Platform feature adoption | float | [0.0, 1.0] | Account data |

## Preprocessing Pipelines

### Text Preprocessing

The text preprocessing pipeline handles text feature transformations:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from within.models.ml.features.transformers import TextCleaner, EntityExtractor

text_preprocessing_pipeline = Pipeline([
    ('cleaner', TextCleaner(
        remove_stopwords=True,
        lowercase=True,
        remove_punctuation=True
    )),
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('vectorizer', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=5
    )),
    ('entity_extractor', EntityExtractor()),
    ('dimensionality_reduction', TruncatedSVD(n_components=100))
])
```

### Numerical Preprocessing

The numerical preprocessing pipeline handles numerical feature transformations:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from within.models.ml.features.transformers import OutlierClipper

numerical_preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier_clipper', OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
    ('scaler', StandardScaler())
])
```

### Categorical Preprocessing

The categorical preprocessing pipeline handles categorical feature transformations:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from within.models.ml.features.transformers import CategoryGrouper

categorical_preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('grouper', CategoryGrouper(min_frequency=0.01, other_category='OTHER')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
```

### Combined Preprocessing

The combined preprocessing pipeline integrates all feature types:

```python
from sklearn.compose import ColumnTransformer

preprocessing_pipeline = ColumnTransformer([
    ('text_features', text_preprocessing_pipeline, [
        'headline', 'description', 'cta_text'
    ]),
    ('numerical_features', numerical_preprocessing_pipeline, [
        'ctr_historical', 'engagement_rate', 'conversion_rate',
        'cost_per_acquisition', 'return_on_ad_spend'
    ]),
    ('categorical_features', categorical_preprocessing_pipeline, [
        'platform', 'industry', 'ad_format', 'targeting_type'
    ])
])
```

## Feature Selection

### Selection Methods

The platform employs multiple feature selection methods:

1. **Filter Methods**:
   - Correlation analysis (remove highly correlated features)
   - Variance thresholding (remove low-variance features)
   - Statistical tests (ANOVA, chi-squared)

2. **Wrapper Methods**:
   - Recursive feature elimination
   - Forward/backward selection
   - Genetic algorithm selection

3. **Embedded Methods**:
   - LASSO regularization
   - Random Forest feature importance
   - Gradient Boosting feature importance

### Implementation

Feature selection is implemented in `app/models/ml/features/selection.py`:

```python
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """Feature selection for ML models"""
    
    def __init__(self, method='importance', params=None):
        """
        Args:
            method: Selection method ('correlation', 'importance', 'rfe')
            params: Parameters for the selection method
        """
        self.method = method
        self.params = params or {}
        self.selected_features = None
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features based on the chosen method
        
        Args:
            X: Feature DataFrame
            y: Target variable
            
        Returns:
            DataFrame with selected features
        """
        if self.method == 'correlation':
            return self._select_by_correlation(X, y)
        elif self.method == 'importance':
            return self._select_by_importance(X, y)
        elif self.method == 'rfe':
            return self._select_by_rfe(X, y)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def _select_by_correlation(self, X, y):
        # Implementation details in selection.py
        pass
    
    def _select_by_importance(self, X, y):
        # Implementation details in selection.py
        pass
    
    def _select_by_rfe(self, X, y):
        # Implementation details in selection.py
        pass
    
    def get_selected_features(self) -> List[str]:
        """Get the list of selected feature names"""
        return self.selected_features
```

## Feature Engineering Guidelines

### Best Practices

1. **Domain Knowledge Integration**:
   - Consult domain experts when defining features
   - Incorporate industry-specific knowledge into feature definitions

2. **Feature Quality**:
   - Ensure features have clear semantic meaning
   - Validate feature distributions and correlations
   - Document feature limitations and edge cases

3. **Efficiency**:
   - Optimize computation for expensive features
   - Cache intermediate results when appropriate
   - Use batch processing for computationally intensive features

4. **Maintainability**:
   - Keep feature definitions in centralized location
   - Document extraction logic thoroughly
   - Use consistent naming conventions

5. **Testing**:
   - Create unit tests for feature extractors
   - Validate feature distributions against expectations
   - Test edge cases and boundary conditions

### Feature Naming Convention

Features follow this naming convention:

```
{category}_{subcategory}_{descriptor}_{statistic}
```

Examples:
- `text_sentiment_positivity_score`
- `visual_color_contrast_mean`
- `perf_engagement_ctr_30d_avg`

### Documentation Template

```markdown
## Feature: {feature_name}

### Definition
{clear definition of what the feature represents}

### Calculation
{formula or procedure used to calculate the feature}

### Input Requirements
{required inputs for calculation}

### Range and Distribution
- **Type**: {data type}
- **Range**: {theoretical range of values}
- **Typical Range**: {typical range of values in practice}
- **Distribution**: {normal, log-normal, etc.}

### Use Cases
{primary use cases for this feature}

### Limitations
{known limitations or edge cases}

### Dependencies
{dependencies on other features or data sources}

### Version History
{changes over time to feature calculation}
```

## Feature Store Integration

### Feature Store Architecture

The feature store architecture includes:

1. **Online Store**: Low-latency access for serving features
2. **Offline Store**: High-throughput access for training
3. **Feature Registry**: Metadata and discovery service
4. **Feature Pipelines**: Scheduled processing jobs
5. **Monitoring**: Feature quality and drift detection

### Feature Store API

```python
from within.ml.features import FeatureStore

# Initialize feature store
feature_store = FeatureStore()

# Get feature for entity
features = feature_store.get_features(
    feature_names=["text_embedding_headline", "visual_complexity"],
    entity_id="ad_123456",
    entity_type="ad"
)

# Get historical features
historical_features = feature_store.get_historical_features(
    feature_names=["ctr_historical"],
    entity_id="ad_123456",
    entity_type="ad",
    start_time="2023-01-01",
    end_time="2023-01-31"
)

# Register new feature
feature_store.register_feature(
    name="new_feature",
    entity_types=["ad"],
    description="Description of new feature",
    owner="team@within.co",
    data_type="float",
    category="performance"
)

# Create feature group
feature_store.create_feature_group(
    name="engagement_features",
    features=["ctr_historical", "engagement_rate", "bounce_rate"],
    description="Features related to user engagement"
)
```

## Feature Monitoring

### Metrics Monitored

The following metrics are monitored for all features:

1. **Distribution Metrics**:
   - Mean, median, standard deviation
   - Quantiles (p1, p5, p25, p50, p75, p95, p99)
   - Min/max values

2. **Quality Metrics**:
   - Missing value rate
   - Null/zero value rate
   - Cardinality (for categorical features)

3. **Drift Metrics**:
   - Kolmogorov-Smirnov test
   - Population Stability Index (PSI)
   - Jensen-Shannon divergence

### Alerting Thresholds

Alerts are triggered when these thresholds are exceeded:

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| Missing Value Rate | >5% increase | >20% increase |
| Mean Drift | >2 standard deviations | >4 standard deviations |
| PSI | >0.1 | >0.2 |
| KS-test p-value | <0.05 | <0.01 |

### Monitoring Implementation

Feature monitoring is implemented in `app/monitoring/feature_monitor.py`:

```python
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from within.monitoring.base import Monitor
from within.monitoring.alerting import AlertManager

class FeatureMonitor(Monitor):
    """Monitor feature distributions and quality"""
    
    def __init__(self, features: List[str], 
                 baseline_period: str = "7d",
                 alert_manager: Optional[AlertManager] = None):
        """
        Args:
            features: List of features to monitor
            baseline_period: Period for baseline calculation
            alert_manager: Alert manager for sending alerts
        """
        self.features = features
        self.baseline_period = baseline_period
        self.alert_manager = alert_manager or AlertManager()
        self.baseline_stats = {}
        
    def compute_baseline(self, data: pd.DataFrame):
        """Compute baseline statistics for features
        
        Args:
            data: DataFrame containing feature data
        """
        # Implementation details in feature_monitor.py
        pass
        
    def check_drift(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Check for feature drift
        
        Args:
            data: DataFrame containing current feature data
            
        Returns:
            Dictionary with drift metrics for each feature
        """
        # Implementation details in feature_monitor.py
        pass
        
    def alert_if_needed(self, drift_metrics: Dict[str, Dict]):
        """Send alerts if drift metrics exceed thresholds
        
        Args:
            drift_metrics: Dictionary with drift metrics for each feature
        """
        # Implementation details in feature_monitor.py
        pass
```

## Versioning and Compatibility

### Feature Versioning

Features are versioned using semantic versioning:

- **MAJOR**: Breaking changes to feature definition or calculation
- **MINOR**: Backward-compatible enhancements to feature
- **PATCH**: Bug fixes or minor improvements

### Compatibility Matrix

The compatibility matrix shows which feature versions are compatible with which model versions:

| Feature | v1.0.x | v1.1.x | v2.0.x |
|---------|--------|--------|--------|
| Ad Score Predictor v1.0.x | ✓ | ✓ | ✗ |
| Ad Score Predictor v2.0.x | ✗ | ✓ | ✓ |
| Account Health v1.0.x | ✓ | ✓ | ✗ |
| Account Health v2.0.x | ✗ | ✓ | ✓ |

### Migration Guide

When updating features, follow these guidelines:

1. **Backward Compatibility**: Maintain backward compatibility when possible
2. **Dual Operation**: Run old and new versions in parallel during transition
3. **Validation**: Validate new features against previous versions
4. **Documentation**: Document changes and migration paths
5. **Deprecation Policy**: Announce deprecation at least one release cycle in advance

## Feature Development Process

### Development Lifecycle

The feature development lifecycle consists of:

1. **Ideation**: Propose new feature based on business need or insight
2. **Specification**: Define feature requirements and calculation methodology
3. **Implementation**: Implement feature extraction and transformation
4. **Testing**: Validate feature implementation and distribution
5. **Review**: Code review and quality assessment
6. **Deployment**: Deploy feature to production
7. **Monitoring**: Monitor feature in production

### Review Checklist

Feature reviews should check:

- [ ] Feature has clear business justification
- [ ] Feature implementation matches specification
- [ ] Feature has appropriate unit tests
- [ ] Feature distribution meets expectations
- [ ] Feature has proper documentation
- [ ] Feature computation is efficient
- [ ] Feature dependencies are explicit
- [ ] Feature has appropriate error handling

### Example: Developing a New Feature

```python
# 1. Define feature in feature registry
from within.ml.features import FeatureRegistry

registry = FeatureRegistry()
registry.register(
    name="ad_text_emotional_impact",
    version="1.0.0",
    description="Measures emotional impact of ad text",
    owner="nlp-team@within.co",
    category="text",
    data_type="float",
    range=[0.0, 100.0]
)

# 2. Implement feature extraction
from within.ml.features.extractors import BaseFeatureExtractor

class EmotionalImpactExtractor(BaseFeatureExtractor):
    """Extract emotional impact score from ad text"""
    
    def __init__(self):
        super().__init__(features=["ad_text_emotional_impact"])
        self.emotion_model = load_emotion_model()
    
    def extract(self, text_data):
        """Extract emotional impact from text
        
        Args:
            text_data: Dictionary with text fields
            
        Returns:
            Dictionary with emotional impact feature
        """
        combined_text = self._combine_text(text_data)
        emotion_scores = self.emotion_model.predict(combined_text)
        impact_score = self._calculate_impact(emotion_scores)
        
        return {"ad_text_emotional_impact": impact_score}
    
    def _combine_text(self, text_data):
        # Combine text fields
        pass
    
    def _calculate_impact(self, emotion_scores):
        # Calculate impact score from emotion scores
        pass

# 3. Register the extractor
from within.ml.features import ExtractorRegistry

extractor_registry = ExtractorRegistry()
extractor_registry.register(
    name="emotional_impact",
    extractor=EmotionalImpactExtractor(),
    features=["ad_text_emotional_impact"]
)
```

---

**Document Revision History**:
- v1.0 (2023-01-05): Initial feature documentation
- v1.1 (2023-03-10): Added feature monitoring section
- v1.2 (2023-06-20): Updated feature store integration
- v2.0 (2023-09-15): Major revision with complete feature catalog 
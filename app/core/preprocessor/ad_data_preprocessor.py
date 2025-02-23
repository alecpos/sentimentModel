# ad_data_preprocessor.py
import pandas as pd
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced NLP preprocessing with spaCy and VADER sentiment analysis"""
    def __init__(self, nlp_model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(nlp_model)
        self.sentiment = SentimentIntensityAnalyzer()
        
    def _tokenize(self, text: str) -> List[str]:
        """Spacy-based tokenization with lemmatization"""
        doc = self.nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """VADER sentiment analysis with compound score"""
        return self.sentiment.polarity_scores(text)

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        """Transform text data with NLP features"""
        logger.info("Processing text features...")
        processed = X.apply(lambda text: self._process_text(text))
        return pd.DataFrame(list(processed))

    def _process_text(self, text: str) -> Dict[str, Any]:
        """Process individual text entry"""
        if pd.isnull(text):
            return {
                'tokens': [],
                'sentiment_compound': 0.0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'char_count': 0,
                'word_count': 0
            }
            
        tokens = self._tokenize(text)
        sentiment = self._analyze_sentiment(text)
        
        return {
            'tokens': tokens,
            'sentiment_compound': sentiment['compound'],
            'sentiment_positive': sentiment['pos'],
            'sentiment_negative': sentiment['neg'],
            'char_count': len(text),
            'word_count': len(text.split())
        }

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Robust outlier handling with IQR-based clipping"""
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        """Fit the outlier handler.
        
        Args:
            X: array-like or DataFrame of shape (n_samples, n_features)
            y: Ignored
            
        Returns:
            self
        """
        # Convert to DataFrame if numpy array
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        
        for col in range(X_df.shape[1]):
            series = X_df.iloc[:, col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            self.bounds[col] = (
                q1 - self.factor * iqr,
                q3 + self.factor * iqr
            )
        return self

    def transform(self, X) -> np.ndarray:
        """Transform the data by clipping outliers.
        
        Args:
            X: array-like or DataFrame of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Transformed array with outliers clipped
        """
        # Convert to DataFrame if numpy array
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        
        for col in range(X_df.shape[1]):
            X_df.iloc[:, col] = X_df.iloc[:, col].clip(
                lower=self.bounds[col][0],
                upper=self.bounds[col][1]
            )
        
        return X_df.to_numpy()

class DataValidator(BaseEstimator, TransformerMixin):
    """Post-processing validation transformer"""
    def __init__(self, expected_schema: Dict[str, type]):
        self.expected_schema = expected_schema
        self.feature_names = list(expected_schema.keys())

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        """Transform and validate the data.
        
        Args:
            X: array-like or DataFrame of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Validated array
            
        Raises:
            ValueError: If validation fails
        """
        # Convert to DataFrame with feature names if numpy array
        if isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
                )
            X = pd.DataFrame(X, columns=self.feature_names)
        
        self._validate_data(X)
        return X.to_numpy()

    def _validate_data(self, X: pd.DataFrame):
        """Ensure data meets quality standards"""
        # Schema validation
        for col, dtype in self.expected_schema.items():
            if col not in X.columns:
                raise ValueError(f"Missing expected column: {col}")
            if not np.issubdtype(X[col].dtype, dtype):
                try:
                    X[col] = X[col].astype(dtype)
                except:
                    raise TypeError(
                        f"Invalid dtype for {col}. Expected {dtype}, got {X[col].dtype}"
                    )
        
        # Check for remaining missing values
        if X.isnull().sum().sum() > 0:
            missing = X.isnull().sum()
            raise ValueError(f"Missing values detected:\n{missing[missing > 0]}")
        
        # Validate sentiment scores if present
        sentiment_cols = [c for c in X.columns if 'sentiment' in c]
        for col in sentiment_cols:
            if (X[col] < -1).any() or (X[col] > 1).any():
                raise ValueError(
                    f"Invalid values in {col}. Sentiment scores must be between -1 and 1"
                )

# ad_data_preprocessor.py (updated)
def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    text_features: List[str],
    config: Dict[str, Any]
) -> Pipeline:
    """Flexible preprocessing pipeline builder with enhanced validation.
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        text_features: List of text column names
        config: Configuration dictionary with preprocessing options
        
    Returns:
        Pipeline: Scikit-learn pipeline for data preprocessing
        
    Raises:
        ValueError: If feature lists overlap or required config is missing
    """
    # Validate feature lists
    all_features = set(numerical_features + categorical_features + text_features)
    if len(all_features) != len(numerical_features) + len(categorical_features) + len(text_features):
        raise ValueError("Feature lists must not overlap")
        
    # Define output schema dynamically
    output_schema = {
        **{col: np.float64 for col in numerical_features},
        **{col: np.object_ for col in categorical_features}
    }
    
    # Add text-derived features to schema if text features present
    if text_features:
        output_schema.update({
            'sentiment_compound': np.float64,
            'sentiment_positive': np.float64,
            'sentiment_negative': np.float64,
            'char_count': np.int64,
            'word_count': np.int64
        })

    # Build transformers list
    transformers = []
    
    # Add numerical pipeline if features present
    if numerical_features:
        transformers.append((
            'numerical', 
            Pipeline([
                ('imputer', SimpleImputer(strategy=config.get('numerical_impute_strategy', 'median'))),
                ('outlier', OutlierHandler(factor=config.get('outlier_factor', 1.5))),
                ('scaler', StandardScaler())
            ]), 
            numerical_features
        ))
    
    # Add categorical pipeline if features present
    if categorical_features:
        transformers.append((
            'categorical',
            Pipeline([
                ('imputer', SimpleImputer(strategy=config.get('categorical_impute_strategy', 'most_frequent'))),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]),
            categorical_features
        ))
    
    # Add text pipeline if features present
    if text_features:
        text_config = config.get('text_features', {})
        transformers.append((
            'text',
            Pipeline([
                ('processor', TextPreprocessor(
                    nlp_model=text_config.get('nlp_model', 'en_core_web_sm')
                )),
                ('vectorizer', ColumnTransformer([
                    ('tokens', TfidfVectorizer(
                        max_features=text_config.get('max_features', 500),
                        tokenizer=lambda x: x,
                        preprocessor=lambda x: x
                    ), 'tokens'),
                    ('sentiment', 'passthrough', [
                        'sentiment_compound',
                        'sentiment_positive',
                        'sentiment_negative'
                    ]),
                    ('counts', 'passthrough', [
                        'char_count',
                        'word_count'
                    ])
                ]))
            ]),
            text_features
        ))

    # Return complete pipeline
    return Pipeline([
        ('feature_engineering', ColumnTransformer(
            transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )),
        ('validation', DataValidator(output_schema))
    ])

# Example usage:
if __name__ == "__main__":
    config = {
        'text_features': {
            'max_features': 500,
            'nlp_model': 'en_core_web_sm'
        }
    }
    
    pipeline = build_preprocessing_pipeline(config)
    
    # Sample data that matches test schemas
    sample_data = pd.DataFrame({
        'impressions': [1000, 2000, None],
        'clicks': [50, 200, 300],
        'spend': [500.0, 2000.0, 1500.0],
        'conversions': [10, 50, None],
        'campaign_type': ['search', 'display', None],
        'platform': ['google', 'facebook', 'instagram'],
        'target_audience': ['retargeting', 'prospecting', 'lookalike'],
        'ad_copy': ['Great deals today!', None, 'Limited time offer'],
        'landing_page_text': ['Free shipping available', '50% off sale', '']
    })
    
    processed = pipeline.fit_transform(sample_data)
    print("Processed data shape:", processed.shape)
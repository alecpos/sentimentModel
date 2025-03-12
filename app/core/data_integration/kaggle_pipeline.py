"""
Kaggle Dataset Integration Pipeline for WITHIN System

This module provides a robust pipeline for integrating Kaggle datasets into the
WITHIN Ad Score & Account Health Predictor system. It handles data downloading,
validation, preprocessing, and harmonization to ensure consistent inputs for 
model training and evaluation, with a focus on fairness considerations.

Implementation follows WITHIN ML Backend standards with strict type safety,
validation requirements, and fairness evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import hashlib
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from shutil import copyfile
import requests
import tempfile
import zipfile
import kaggle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error

from app.core.validation import DataValidationSchema, validate_dataset
from app.core.fairness import FairnessEvaluator, FairnessResults
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
from app.models.ml.prediction.account_health_predictor import AccountHealthPredictor
from app.models.ml.prediction.ad_sentiment_analyzer import AdSentimentAnalyzer

logger = logging.getLogger(__name__)

class DatasetCategory(str, Enum):
    """Categories of datasets used in the WITHIN system."""
    
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CUSTOMER_CONVERSION = "customer_conversion"
    CTR_PREDICTION = "ctr_prediction"
    AD_PERFORMANCE = "ad_performance"
    ACCOUNT_HEALTH = "account_health"

@dataclass
class DatasetConfig:
    """Configuration for a Kaggle dataset.
    
    Attributes:
        dataset_slug: Kaggle dataset identifier (username/dataset-name)
        category: Category of the dataset
        target_column: Name of the target column
        feature_columns: List of feature columns to use
        protected_attributes: List of columns containing protected attributes
        schema_path: Path to the validation schema
        preprocessing_fn: Optional preprocessing function
        validation_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        file_patterns: List of file patterns to download
    """
    
    dataset_slug: str
    category: DatasetCategory
    target_column: str
    feature_columns: List[str]
    protected_attributes: List[str]
    schema_path: Path
    preprocessing_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    file_patterns: List[str] = field(default_factory=lambda: ["*.csv"])
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.validation_size + self.test_size >= 1.0:
            raise ValueError("Sum of validation_size and test_size must be less than 1.0")
        
        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

@dataclass
class DatasetMetadata:
    """Metadata for a processed dataset.
    
    Attributes:
        dataset_name: Name of the dataset
        config: Original dataset configuration
        data_hash: Hash of the raw dataset
        processing_timestamp: When the data was processed
        feature_statistics: Statistical properties of features
        fairness_metrics: Fairness evaluation results
        data_drift_metrics: Metrics for data drift detection
        rows_count: Number of rows in the dataset
        columns_count: Number of columns in the dataset
        missing_values_count: Count of missing values
        outliers_count: Count of outliers detected
    """
    
    dataset_name: str
    config: DatasetConfig
    data_hash: str
    processing_timestamp: datetime = field(default_factory=datetime.now)
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fairness_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    data_drift_metrics: Dict[str, float] = field(default_factory=dict)
    rows_count: int = 0
    columns_count: int = 0
    missing_values_count: int = 0
    outliers_count: int = 0

@dataclass
class ProcessedDataset:
    """A processed dataset ready for model training.
    
    Attributes:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        metadata: Dataset metadata
        protected_attributes: Protected attributes for fairness evaluation
    """
    
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: DatasetMetadata
    protected_attributes: Dict[str, pd.Series] = field(default_factory=dict)

class KaggleDatasetPipeline:
    """Pipeline for downloading, processing, and validating Kaggle datasets."""
    
    def __init__(
        self, 
        data_dir: str = "data/kaggle", 
        cache_dir: str = "data/cache",
        validate_fairness: bool = True
    ) -> None:
        """Initialize the Kaggle dataset pipeline.
        
        Args:
            data_dir: Directory to store downloaded datasets
            cache_dir: Directory to store cached preprocessed datasets
            validate_fairness: Whether to validate fairness of datasets
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.validate_fairness = validate_fairness
        self.fairness_evaluator = FairnessEvaluator()
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate Kaggle API credentials
        self._validate_kaggle_credentials()
        
        logger.info(f"Initialized Kaggle Dataset Pipeline with data_dir={data_dir}")

    def _validate_kaggle_credentials(self) -> None:
        """Validate Kaggle API credentials."""
        try:
            # Initialize the Kaggle API
            kaggle.api.authenticate()
            
            # Check if credentials are properly set
            if not (kaggle.api.get_config_value('username') and 
                    kaggle.api.get_config_value('key')):
                if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
                    logger.warning(
                        "Kaggle API credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY "
                        "environment variables or create ~/.kaggle/kaggle.json"
                    )
                    return
            
            # Test API connection by listing datasets (without using the 'limit' parameter)
            kaggle.api.dataset_list(search="test")
            logger.info("Kaggle API credentials validated successfully")
        
        except Exception as e:
            logger.error(f"Failed to validate Kaggle credentials: {str(e)}")
            raise RuntimeError(f"Kaggle API credentials validation failed: {str(e)}")

    def download_dataset(self, config: DatasetConfig) -> Path:
        """Download a dataset from Kaggle.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Path to the downloaded dataset directory
        """
        dataset_dir = self.data_dir / config.dataset_slug.replace("/", "_")
        
        # Check if dataset already exists
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            logger.info(f"Dataset already downloaded: {config.dataset_slug}")
            return dataset_dir
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading dataset: {config.dataset_slug}")
            kaggle.api.dataset_download_files(
                config.dataset_slug,
                path=str(dataset_dir),
                unzip=True
            )
            logger.info(f"Successfully downloaded dataset: {config.dataset_slug}")
            return dataset_dir
        
        except Exception as e:
            logger.error(f"Failed to download dataset {config.dataset_slug}: {str(e)}")
            raise RuntimeError(f"Dataset download failed: {str(e)}")

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the dataframe for caching and tracking.
        
        Args:
            df: Dataframe to hash
            
        Returns:
            Hash string
        """
        # Create a string representation of the dataframe and hash it
        df_str = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.md5(df_str).hexdigest()

    def _validate_data(self, df: pd.DataFrame, schema_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate a dataframe against a schema.
        
        Args:
            df: Dataframe to validate
            schema_path: Path to schema file
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        try:
            schema = DataValidationSchema.from_json(schema_path)
            validation_result = validate_dataset(df, schema)
            is_valid = validation_result.is_valid
            
            if not is_valid:
                logger.warning(
                    f"Data validation failed: {len(validation_result.issues)} issues found. "
                    f"Schema: {schema_path}"
                )
            else:
                logger.info(f"Data validation successful: {schema_path}")
                
            return is_valid, validation_result.to_dict()
        
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False, {"is_valid": False, "errors": [str(e)]}

    def _compute_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute statistical properties of features for metadata.
        
        Args:
            df: Dataframe to analyze
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {}
        
        for col in df.columns:
            col_stats = {}
            
            # Skip non-numeric columns for certain statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "skew": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis())
                })
            
            # Common statistics for all columns
            col_stats.update({
                "missing": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean()),
                "unique_values": int(df[col].nunique())
            })
            
            stats[col] = col_stats
            
        return stats

    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[int, pd.DataFrame]:
        """Detect outliers in numeric columns using IQR method.
        
        Args:
            df: Dataframe to analyze
            numeric_cols: List of numeric columns to check
            
        Returns:
            Tuple of (outliers_count, outlier_mask)
        """
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Count total number of rows with at least one outlier
        rows_with_outliers = outlier_mask.any(axis=1).sum()
        
        return rows_with_outliers, outlier_mask

    def _evaluate_fairness(
        self, 
        df: pd.DataFrame, 
        protected_attributes: List[str],
        target_column: str
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate fairness metrics for protected attributes.
        
        Args:
            df: Dataframe containing features and target
            protected_attributes: List of columns containing protected attributes
            target_column: Name of the target column
            
        Returns:
            Dictionary of fairness metrics by protected attribute
        """
        if not self.validate_fairness:
            logger.info("Fairness validation skipped")
            return {}
            
        if not protected_attributes:
            logger.info("No protected attributes specified for fairness evaluation")
            return {}
            
        fairness_results = {}
        
        try:
            # Run fairness evaluation for each protected attribute
            for attr in protected_attributes:
                if attr not in df.columns:
                    logger.warning(f"Protected attribute not found in data: {attr}")
                    continue
                    
                # Get unique values of the protected attribute
                groups = df[attr].unique()
                
                # Skip if there's only one group
                if len(groups) <= 1:
                    logger.warning(f"Only one group found for attribute {attr}")
                    continue
                
                # Configure and run fairness evaluation
                fairness_results[attr] = self.fairness_evaluator.evaluate(
                    df=df,
                    protected_attribute=attr,
                    target_column=target_column
                ).to_dict()
                    
            # For intersectional fairness, evaluate pairs of protected attributes
            if len(protected_attributes) >= 2:
                for i, attr1 in enumerate(protected_attributes):
                    for attr2 in protected_attributes[i+1:]:
                        if attr1 in df.columns and attr2 in df.columns:
                            # Create intersectional attribute
                            df[f"{attr1}_{attr2}"] = df[attr1].astype(str) + "_" + df[attr2].astype(str)
                            
                            # Evaluate fairness
                            fairness_results[f"{attr1}_{attr2}"] = self.fairness_evaluator.evaluate(
                                df=df,
                                protected_attribute=f"{attr1}_{attr2}",
                                target_column=target_column
                            ).to_dict()
                            
                            # Remove temporary column
                            df.drop(f"{attr1}_{attr2}", axis=1, inplace=True)
            
            return fairness_results
            
        except Exception as e:
            logger.error(f"Error during fairness evaluation: {str(e)}")
            return {}

    def _load_dataset_files(self, dataset_dir: Path, file_patterns: List[str]) -> pd.DataFrame:
        """Load and combine dataset files matching the provided patterns.
        
        Args:
            dataset_dir: Directory containing dataset files
            file_patterns: List of file patterns to match
            
        Returns:
            Combined dataframe
        """
        all_files = []
        
        # Find all files matching the patterns
        for pattern in file_patterns:
            matching_files = list(dataset_dir.glob(pattern))
            all_files.extend(matching_files)
            
        if not all_files:
            raise FileNotFoundError(f"No files matching patterns {file_patterns} found in {dataset_dir}")
            
        # Load each file and combine them
        dataframes = []
        for file_path in all_files:
            try:
                file_extension = file_path.suffix.lower()
                
                if file_extension == '.csv':
                    # Try loading with utf-8 encoding first
                    try:
                        logger.info(f"Attempting to load {file_path} with utf-8 encoding")
                        df = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        # If utf-8 fails, try latin1 which is more permissive
                        logger.info(f"Falling back to latin1 encoding for {file_path}")
                        df = pd.read_csv(file_path, encoding='latin1')
                elif file_extension in ['.xls', '.xlsx']:
                    df = pd.read_excel(file_path)
                elif file_extension == '.json':
                    df = pd.read_json(file_path)
                elif file_extension == '.parquet':
                    df = pd.read_parquet(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue
                    
                dataframes.append(df)
                logger.info(f"Loaded file: {file_path} with {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                raise RuntimeError(f"Failed to load dataset file: {str(e)}")
                
        # Combine all dataframes
        if len(dataframes) == 1:
            return dataframes[0]
        elif len(dataframes) > 1:
            # Check if dataframes can be concatenated
            try:
                return pd.concat(dataframes, ignore_index=True)
            except Exception as e:
                logger.error(f"Error combining dataframes: {str(e)}")
                raise RuntimeError(f"Failed to combine dataset files: {str(e)}")
        else:
            raise ValueError("No dataframes were successfully loaded")

    def process_dataset(self, config: DatasetConfig) -> ProcessedDataset:
        """Process a dataset from Kaggle for use in the WITHIN system.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Processed dataset ready for model training
        """
        # Download dataset if needed
        dataset_dir = self.download_dataset(config)
        
        # Check cache
        cache_key = f"{config.dataset_slug.replace('/', '_')}_{config.random_state}"
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                logger.info(f"Loading cached dataset: {cache_path}")
                processed_dataset = pd.read_pickle(cache_path)
                return processed_dataset
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {str(e)}")
        
        # Load dataset files
        df = self._load_dataset_files(dataset_dir, config.file_patterns)
        
        # Special handling for Sentiment140 dataset
        if config.dataset_slug == "kazanova/sentiment140":
            logger.info("Applying special preprocessing for Sentiment140 dataset")
            
            # The Sentiment140 dataset has no header, columns are:
            # 0: target (0 = negative, 4 = positive)
            # 1: id
            # 2: date
            # 3: query
            # 4: user
            # 5: text
            
            column_names = ['target', 'id', 'date', 'query', 'user', 'text']
            df.columns = column_names
            
            # Convert target from 0/4 to 0/1
            df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
            
            logger.info(f"Processed Sentiment140 dataset: {len(df)} rows, target distribution: {df['target'].value_counts().to_dict()}")
        
        # Compute data hash for tracking
        data_hash = self._compute_data_hash(df)
        
        # Apply preprocessing if specified
        if config.preprocessing_fn is not None:
            try:
                logger.info("Applying custom preprocessing function")
                df = config.preprocessing_fn(df)
            except Exception as e:
                logger.error(f"Error in preprocessing function: {str(e)}")
                raise RuntimeError(f"Preprocessing failed: {str(e)}")
        
        # Validate against schema
        is_valid, validation_results = self._validate_data(df, config.schema_path)
        if not is_valid:
            logger.error("Dataset failed validation - attempting to continue with warnings")
        
        # Compute feature statistics and detect outliers
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_statistics = self._compute_feature_statistics(df)
        outliers_count, outlier_mask = self._detect_outliers(df, numeric_cols)
        
        # Verify that required columns exist
        missing_cols = []
        if config.target_column not in df.columns:
            missing_cols.append(config.target_column)
        
        for col in config.feature_columns:
            if col not in df.columns:
                missing_cols.append(col)
                
        if missing_cols:
            raise ValueError(f"Required columns not found in dataset: {missing_cols}")
        
        # Extract features and target
        X = df[config.feature_columns]
        y = df[config.target_column]
        
        # Extract protected attributes for fairness evaluation
        protected_attributes_data = {}
        for attr in config.protected_attributes:
            if attr in df.columns:
                protected_attributes_data[attr] = df[attr]
        
        # Evaluate fairness on the entire dataset
        fairness_metrics = self._evaluate_fairness(
            df, config.protected_attributes, config.target_column
        )
        
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        test_size_adjusted = config.validation_size / (1 - config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=test_size_adjusted, 
            random_state=config.random_state
        )
        
        # Create dataset metadata
        metadata = DatasetMetadata(
            dataset_name=config.dataset_slug.split('/')[-1],
            config=config,
            data_hash=data_hash,
            feature_statistics=feature_statistics,
            fairness_metrics=fairness_metrics,
            rows_count=len(df),
            columns_count=len(df.columns),
            missing_values_count=df.isna().sum().sum(),
            outliers_count=outliers_count
        )
        
        # Create processed dataset
        processed_dataset = ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata,
            protected_attributes={
                attr: df[attr] for attr in config.protected_attributes if attr in df.columns
            }
        )
        
        # Cache the processed dataset
        try:
            logger.info(f"Caching processed dataset: {cache_path}")
            pd.to_pickle(processed_dataset, cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache processed dataset: {str(e)}")
        
        return processed_dataset
    
    def get_dataset_configs(self) -> Dict[str, DatasetConfig]:
        """Get predefined dataset configurations for WITHIN system components.
        
        Returns:
            Dictionary of dataset configurations by name
        """
        return {
            "sentiment140": DatasetConfig(
                dataset_slug="kazanova/sentiment140",
                category=DatasetCategory.SENTIMENT_ANALYSIS,
                target_column="target",
                feature_columns=["text"],
                protected_attributes=[],  # Sentiment140 doesn't contain demographic info
                schema_path=Path("app/core/schemas/sentiment140_schema.json"),
                file_patterns=["*.csv"]
            ),
            
            "social_media_sentiments": DatasetConfig(
                dataset_slug="kashishparmar02/social-media-sentiments-analysis-dataset",
                category=DatasetCategory.SENTIMENT_ANALYSIS,
                target_column="sentiment",
                feature_columns=["text", "source"],
                protected_attributes=[],  # No demographic info
                schema_path=Path("app/core/schemas/social_media_sentiments_schema.json"),
                file_patterns=["*.csv"]
            ),
            
            "customer_conversion": DatasetConfig(
                dataset_slug="muhammadshahidazeem/customer-conversion-dataset-for-stuffmart-com",
                category=DatasetCategory.CUSTOMER_CONVERSION,
                target_column="converted",
                feature_columns=[
                    "age", "gender", "location", "time_spent", "pages_visited", 
                    "device", "browser", "previous_visits", "referral_source"
                ],
                protected_attributes=["age", "gender", "location"],
                schema_path=Path("app/core/schemas/customer_conversion_schema.json"),
                file_patterns=["*.csv"]
            ),
            
            "ctr_optimization": DatasetConfig(
                dataset_slug="rahulchavan99/the-click-through-rate-ctr-optimization",
                category=DatasetCategory.CTR_PREDICTION,
                target_column="click",
                feature_columns=[
                    "impression", "campaign_id", "ad_id", "placement", "device_type",
                    "browser", "time_of_day", "day_of_week"
                ],
                protected_attributes=["device_type"],
                schema_path=Path("app/core/schemas/ctr_optimization_schema.json"),
                file_patterns=["*.csv"]
            )
        }
    
    def get_sentiment_datasets(self) -> List[ProcessedDataset]:
        """Get all sentiment analysis datasets.
        
        Returns:
            List of processed sentiment analysis datasets
        """
        configs = self.get_dataset_configs()
        sentiment_configs = [
            config for name, config in configs.items() 
            if config.category == DatasetCategory.SENTIMENT_ANALYSIS
        ]
        
        return [self.process_dataset(config) for config in sentiment_configs]
    
    def get_conversion_datasets(self) -> List[ProcessedDataset]:
        """Get all customer conversion datasets.
        
        Returns:
            List of processed customer conversion datasets
        """
        configs = self.get_dataset_configs()
        conversion_configs = [
            config for name, config in configs.items() 
            if config.category == DatasetCategory.CUSTOMER_CONVERSION
        ]
        
        return [self.process_dataset(config) for config in conversion_configs]
    
    def get_ctr_datasets(self) -> List[ProcessedDataset]:
        """Get all CTR prediction datasets.
        
        Returns:
            List of processed CTR prediction datasets
        """
        configs = self.get_dataset_configs()
        ctr_configs = [
            config for name, config in configs.items() 
            if config.category == DatasetCategory.CTR_PREDICTION
        ]
        
        return [self.process_dataset(config) for config in ctr_configs] 
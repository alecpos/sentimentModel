"""
Data Integration Module for WITHIN

This module provides data integration capabilities for the WITHIN
Ad Score & Account Health Predictor system, with a focus on integrating
external datasets with fairness evaluation.
"""

from app.core.data_integration.kaggle_pipeline import (
    DatasetCategory,
    DatasetConfig,
    DatasetMetadata,
    ProcessedDataset,
    KaggleDatasetPipeline
)

__all__ = [
    'DatasetCategory',
    'DatasetConfig',
    'DatasetMetadata',
    'ProcessedDataset',
    'KaggleDatasetPipeline'
] 
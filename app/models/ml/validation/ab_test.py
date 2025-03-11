from typing import List, Optional, Dict, Any
import uuid
import logging
from datetime import datetime

class ABTestManager:
    """
    Manages A/B testing for ML models, comparing performance between two models
    with real traffic and allowing for statistical analysis of results.
    """
    
    def __init__(
            self, 
            model_a=None,
            model_b=None,
            test_id: Optional[str] = None,
            traffic_split: float = 0.5,
            metrics: List[str] = None,
            significance_level: float = 0.05,
            min_sample_size: int = 1000,
            metadata: Dict[str, Any] = None,
            config: Dict[str, Any] = None
        ):
        """
        Initialize an A/B test manager with two models.
        
        Args:
            model_a: The A model (usually current production model)
            model_b: The B model (usually new candidate model)
            test_id: Unique identifier for the test
            traffic_split: Fraction of traffic to send to model B (0-1)
            metrics: List of metrics to track for comparison
            significance_level: P-value threshold for significance
            min_sample_size: Minimum number of samples needed before analysis
            metadata: Additional metadata about the test
            config: Configuration dictionary (alternative to individual parameters)
        """
        # First check if config is provided, and use its values
        if config is not None:
            self.model_a = model_a
            self.model_b = model_b
            self.test_id = config.get("test_id", f"abtest_{uuid.uuid4().hex[:8]}")
            
            # Get traffic split from the config dictionary
            traffic_split_config = config.get("traffic_split", {})
            if isinstance(traffic_split_config, dict):
                model_b_split = traffic_split_config.get("model_b", 0.5)
                self.traffic_split = min(max(model_b_split, 0.0), 1.0)
            else:
                self.traffic_split = 0.5
                
            self.metrics = config.get("metrics", ["conversion_rate", "revenue_per_user", "engagement"])
            self.significance_level = config.get("significance_level", 0.05)
            self.min_sample_size = config.get("min_sample_size", 1000)
            self.metadata = config.copy()  # Keep the whole config as metadata
            
            # Get stratification settings
            self.stratification = config.get("stratification", {"enabled": False})
        else:
            # Use individual parameters
            if model_a is None:
                raise ValueError("Model A must be provided")
                
            if model_b is None:
                raise ValueError("Model B must be provided")
                
            self.model_a = model_a
            self.model_b = model_b
            self.test_id = test_id or f"abtest_{uuid.uuid4().hex[:8]}"
            self.traffic_split = min(max(traffic_split, 0.0), 1.0)  # Ensure between 0-1
            self.metrics = metrics or ["conversion_rate", "revenue_per_user", "engagement"]
            self.significance_level = significance_level
            self.min_sample_size = min_sample_size
            self.metadata = metadata or {}
            
            # Default stratification settings
            self.stratification = {"enabled": False}
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize tracking data
        self.model_a_predictions = []
        self.model_b_predictions = []
        self.model_a_outcomes = []
        self.model_b_outcomes = []
        self.start_time = datetime.now()
        
        # Store additional attributes
        self.is_active = True
        self.conclusion = None
        
        self.logger.info(f"Initialized A/B test {self.test_id} with {self.traffic_split:.1%} traffic to model B") 
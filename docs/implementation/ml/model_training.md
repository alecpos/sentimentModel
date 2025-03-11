# Model Training Process

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document details the end-to-end training process for the machine learning models used in the WITHIN Ad Score & Account Health Predictor system. It covers data preparation, feature engineering, model selection, hyperparameter optimization, and evaluation methodologies.

## Table of Contents

1. [Training Framework](#training-framework)
2. [Data Preparation](#data-preparation)
3. [Feature Engineering Integration](#feature-engineering-integration)
4. [Model Selection](#model-selection)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Training Process](#training-process)
7. [Validation Strategy](#validation-strategy)
8. [Model Persistence](#model-persistence)
9. [Documentation & Artifact Management](#documentation-artifact-management)
10. [CI/CD Integration](#ci-cd-integration)

## Training Framework

The WITHIN model training framework follows a modular design pattern that enables:

1. **Reproducibility**: All training runs are fully reproducible with fixed random seeds and versioned datasets
2. **Traceability**: Training metadata, parameters, and metrics are tracked in experiment logs
3. **Scalability**: Training pipelines can run on local development environments or distributed cloud infrastructure
4. **Extensibility**: New model architectures can be easily integrated into the existing framework 

## Data Preparation

The data preparation process involves multiple stages to ensure high-quality training data:

### 1. Data Collection

Data is collected from multiple sources:

- Ad performance data from advertising platforms (Facebook, Google, TikTok, etc.)
- Campaign configuration metadata
- Creative assets (ad text, images)
- Conversion and engagement metrics
- Audience targeting information

### 2. Data Validation

Before entering the training pipeline, data undergoes rigorous validation:

```python
def validate_training_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate training data for completeness and correctness."""
    errors = []
    
    # Check for required columns
    required_columns = [
        "ad_id", "platform", "ad_text", "campaign_objective", 
        "target_audience", "spend", "impressions", "clicks", 
        "conversions", "conversion_value", "start_date", "end_date"
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for null values in critical columns
    for col in ["ad_id", "platform", "ad_text", "campaign_objective"]:
        if col in data.columns and data[col].isnull().any():
            null_count = data[col].isnull().sum()
            errors.append(f"Column {col} contains {null_count} null values")
    
    return len(errors) == 0, errors
```

### 3. Data Cleaning

After validation, the data undergoes cleaning:

- Removal of duplicate records
- Handling of missing values
- Correction of data type inconsistencies
- Filtering out outliers
- Normalization of text data 

## Feature Engineering Integration

The model training process integrates with the feature engineering pipeline described in the [Feature Engineering Documentation](/docs/implementation/ml/feature_engineering.md). This section details how the feature engineering components are incorporated into the training workflow.

### 1. Feature Configuration

Features are configured through a YAML file that defines which features to extract, transform, and select:

```yaml
# feature_config.yaml
text_features:
  - name: sentiment_score
    type: nlp
    transformer: SentimentTransformer
    params:
      model_name: "distilbert-base-uncased-finetuned-sst-2-english"
  
  - name: emotion_scores
    type: nlp
    transformer: EmotionTransformer
    params:
      emotions: ["joy", "sadness", "anger", "fear", "surprise"]

  - name: message_length
    type: statistical
    transformer: TextLengthTransformer
    
campaign_features:
  - name: platform
    type: categorical
    transformer: OneHotEncoder
    params:
      categories: ["facebook", "instagram", "google", "tiktok", "pinterest"]
      
  - name: objective
    type: categorical
    transformer: OneHotEncoder
    params:
      categories: ["awareness", "consideration", "conversion"]
```

### 2. Feature Pipeline Construction

The feature pipelines are constructed during the training phase:

```python
def build_feature_pipelines(config_path: str) -> Dict[str, Pipeline]:
    """Build scikit-learn pipelines for feature processing."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    pipelines = {}
    
    # Build text feature pipeline
    text_transformers = []
    for feature_config in config.get("text_features", []):
        transformer_class = getattr(text_transformers_module, feature_config["transformer"])
        transformer = transformer_class(**feature_config.get("params", {}))
        text_transformers.append((feature_config["name"], transformer))
    
    pipelines["text"] = Pipeline(text_transformers)
    
    # Build campaign feature pipeline
    campaign_transformers = []
    for feature_config in config.get("campaign_features", []):
        transformer_class = getattr(campaign_transformers_module, feature_config["transformer"])
        transformer = transformer_class(**feature_config.get("params", {}))
        campaign_transformers.append((feature_config["name"], transformer))
    
    pipelines["campaign"] = Pipeline(campaign_transformers)
    
    return pipelines
```

## Model Selection

The WITHIN system supports multiple model architectures for prediction tasks. This section outlines the model selection process and implemented architectures.

### 1. Model Architecture Options

For Ad Score Prediction, the following model architectures are supported:

#### Gradient Boosting Models

- **XGBoost**: Default model for Ad Score Prediction
- **LightGBM**: Alternative with faster training time
- **CatBoost**: Better handling of categorical features

#### Deep Learning Models

- **Feed-forward Neural Network**: For tabular data
- **Transformer-based Models**: For text-heavy predictions
- **Hybrid Models**: Combining embedding and tabular data

### 2. Model Selection Criteria

Models are selected based on:

1. **Performance Metrics**: Prediction accuracy, RMSE, MAE
2. **Inference Speed**: Latency requirements for production
3. **Interpretability**: Need for feature importance and explanations
4. **Dataset Size**: Some models perform better with more/less data
5. **Feature Types**: Text-heavy vs. tabular data requirements

### 3. Model Selection Implementation

The model selection process is automated using cross-validation:

```python
def select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str = "regression",
    evaluation_metric: str = "rmse",
    n_folds: int = 5
) -> Tuple[str, Any]:
    """Select the best performing model from multiple architectures."""
    # Define model candidates
    models = {
        "xgboost": XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ) if problem_type == "regression" else XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ),
        
        "lightgbm": LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ) if problem_type == "regression" else LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        ),
        
        "neural_network": MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            learning_rate_init=0.001, 
            max_iter=1000, 
            early_stopping=True, 
            random_state=42
        ) if problem_type == "regression" else MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            learning_rate_init=0.001, 
            max_iter=1000, 
            early_stopping=True, 
            random_state=42
        )
    }
    
    # Cross-validation results
    cv_results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(X_train):
            X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_cv_train, y_cv_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            # Train model
            model.fit(X_cv_train, y_cv_train)
            
            # Evaluate
            score = evaluate_model(model, X_cv_test, y_cv_test)
            cv_scores.append(score)
        
        cv_results[name] = {
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "val_score": evaluate_model(model, X_val, y_val)
        }
        
    # Select best model based on validation score
    best_model_name = min(cv_results, key=lambda x: cv_results[x]["val_score"])
    
    # Retrain best model on full training data
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    
    return best_model_name, best_model
```

## Hyperparameter Optimization

After model selection, hyperparameters are optimized to improve performance.

### 1. Hyperparameter Search Strategies

The system implements several search strategies:

- **Grid Search**: Exhaustive search over specified parameter values
- **Random Search**: Random combinations of parameter values
- **Bayesian Optimization**: Sequential optimization with Gaussian Processes
- **Hyperband**: Multi-fidelity optimization for deep learning models

### 2. Hyperparameter Optimization Implementation

The hyperparameter optimization process is implemented using scikit-learn and Optuna:

```python
def optimize_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str = "regression",
    evaluation_metric: str = "rmse",
    n_trials: int = 50,
    timeout: Optional[int] = 3600
) -> Tuple[Dict[str, Any], Any]:
    """Optimize hyperparameters for the selected model."""
    # Define objective function for optuna
    def objective(trial):
        if model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "random_state": 42
            }
            
            if problem_type == "regression":
                model = XGBRegressor(**params)
            else:
                model = XGBClassifier(**params)
                
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "random_state": 42
            }
            
            if problem_type == "regression":
                model = LGBMRegressor(**params)
            else:
                model = LGBMClassifier(**params)
                
        elif model_name == "neural_network":
            params = {
                "hidden_layer_sizes": (
                    trial.suggest_int("hidden_layer_1", 16, 256),
                    trial.suggest_int("hidden_layer_2", 16, 128)
                ),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-3, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                "max_iter": 1000,
                "early_stopping": True,
                "random_state": 42
            }
            
            if problem_type == "regression":
                model = MLPRegressor(**params)
            else:
                model = MLPClassifier(**params)
        
        # Train and evaluate model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        if problem_type == "regression":
            y_pred = model.predict(X_val)
            if evaluation_metric == "rmse":
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                return score  # Lower is better
            elif evaluation_metric == "mae":
                score = mean_absolute_error(y_val, y_pred)
                return score  # Lower is better
        else:  # classification
            y_pred = model.predict_proba(X_val)[:, 1]
            if evaluation_metric == "auc":
                score = roc_auc_score(y_val, y_pred)
                return -score  # Optuna minimizes, so we negate
    
    # Create optuna study
    if problem_type == "regression" and evaluation_metric in ["rmse", "mae"]:
        # Lower is better, so minimize
        study = optuna.create_study(direction="minimize")
    else:
        # Higher is better, so maximize (but objective returns negative values)
        study = optuna.create_study(direction="minimize")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    
    # Create and train final model with best parameters
    if model_name == "xgboost":
        if problem_type == "regression":
            final_model = XGBRegressor(
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", 5),
                learning_rate=best_params.get("learning_rate", 0.1),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 1),
                gamma=best_params.get("gamma", 0),
                random_state=42
            )
        else:
            final_model = XGBClassifier(
                n_estimators=best_params.get("n_estimators", 100),
                max_depth=best_params.get("max_depth", 5),
                learning_rate=best_params.get("learning_rate", 0.1),
                subsample=best_params.get("subsample", 0.8),
                colsample_bytree=best_params.get("colsample_bytree", 0.8),
                min_child_weight=best_params.get("min_child_weight", 1),
                gamma=best_params.get("gamma", 0),
                random_state=42
            )
    
    # Train final model
    final_model.fit(X_train, y_train)
    
    return best_params, final_model
```

### 3. Learning Curves Analysis

Learning curves are generated to diagnose overfitting or underfitting:

```python
def plot_learning_curves(
    model, 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    problem_type: str = "regression",
    evaluation_metric: str = "rmse",
    n_subsets: int = 10
) -> None:
    """Plot learning curves to diagnose overfitting/underfitting."""
    train_sizes = np.linspace(0.1, 1.0, n_subsets)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        # Sample subset of training data
        n_samples = int(X_train.shape[0] * size)
        idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
        X_subset = X_train.iloc[idx]
        y_subset = y_train.iloc[idx]
        
        # Clone and train model
        subset_model = clone(model)
        subset_model.fit(X_subset, y_subset)
        
        # Evaluate on training subset
        if problem_type == "regression":
            train_pred = subset_model.predict(X_subset)
            if evaluation_metric == "rmse":
                train_score = np.sqrt(mean_squared_error(y_subset, train_pred))
            
            # Evaluate on validation set
            val_pred = subset_model.predict(X_val)
            if evaluation_metric == "rmse":
                val_score = np.sqrt(mean_squared_error(y_val, val_pred))
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label=f'Training {evaluation_metric}')
    plt.plot(train_sizes, val_scores, 'o-', label=f'Validation {evaluation_metric}')
    plt.xlabel('Training Set Size Fraction')
    plt.ylabel(evaluation_metric.upper())
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"learning_curves_{evaluation_metric}.png")
    plt.close()
```

## Training Process

The actual model training process combines the previous components into a comprehensive workflow:

1. Load and validate training data
2. Split data into training, validation, and test sets
3. Construct feature pipelines
4. Extract features for all datasets
5. Select baseline model architecture
6. Optimize hyperparameters
7. Train final model
8. Evaluate model performance
9. Calibrate model outputs
10. Persist trained model and metadata

## Validation Strategy

Model validation ensures that trained models generalize well to new data and meet performance requirements.

### 1. Cross-Validation Approach

The system implements multiple cross-validation strategies:

- **K-fold Cross-validation**: For general model validation
- **Stratified K-fold**: For imbalanced classification problems
- **Time-series Cross-validation**: For temporal data with sequential dependencies

```python
def perform_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: Any,
    model_params: Dict[str, Any],
    cv_strategy: str = "kfold",
    n_splits: int = 5,
    problem_type: str = "regression",
    evaluation_metrics: List[str] = ["rmse"],
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform cross-validation with specified strategy."""
    results = {metric: [] for metric in evaluation_metrics}
    
    # Define cross-validation strategy
    if cv_strategy == "kfold":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "stratified":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "timeseries":
        cv = TimeSeriesSplit(n_splits=n_splits)
    
    # Perform cross-validation
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize and train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Generate predictions
        if problem_type == "regression":
            y_pred = model.predict(X_test)
            
            # Compute metrics
            for metric in evaluation_metrics:
                if metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                elif metric == "mae":
                    score = mean_absolute_error(y_test, y_pred)
                elif metric == "r2":
                    score = r2_score(y_test, y_pred)
                
                results[metric].append(score)
    
    # Compute mean and std for each metric
    cv_results = {}
    for metric in evaluation_metrics:
        cv_results[f"{metric}_mean"] = np.mean(results[metric])
        cv_results[f"{metric}_std"] = np.std(results[metric])
    
    return cv_results
```

### 2. Model Fairness Validation

To ensure model fairness across different segments:

```python
def validate_model_fairness(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_feature: str,
    problem_type: str = "regression",
    fairness_metrics: List[str] = ["demographic_parity", "equal_opportunity"]
) -> Dict[str, Any]:
    """Validate model fairness across different segments."""
    # Extract sensitive attribute
    sensitive_values = X[sensitive_feature].unique()
    
    # Generate predictions
    if problem_type == "regression":
        y_pred = model.predict(X)
    else:  # classification
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
    
    # Compute fairness metrics
    fairness_results = {}
    
    for metric in fairness_metrics:
        if metric == "demographic_parity":
            if problem_type == "regression":
                # For regression, compare mean predictions
                group_means = {}
                for value in sensitive_values:
                    mask = X[sensitive_feature] == value
                    group_means[value] = np.mean(y_pred[mask])
                
                # Compute max difference
                values = list(group_means.values())
                max_diff = max(values) - min(values)
                fairness_results["demographic_parity_diff"] = max_diff
    
    return fairness_results
```

## Model Persistence

Trained models and related artifacts are persisted for deployment:

### 1. Saving Model Artifacts

```python
def save_model_artifacts(
    model,
    feature_pipelines: Dict[str, Pipeline],
    calibrator: Any,
    model_name: str,
    version: str,
    output_dir: str,
    metadata: Dict[str, Any]
) -> None:
    """Save model artifacts for deployment."""
    # Create version directory
    model_dir = f"{output_dir}/{model_name}/{version}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    if isinstance(model, xgb.Booster):
        model_path = f"{model_dir}/model.json"
        model.save_model(model_path)
    else:
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    
    # Save feature pipelines
    with open(f"{model_dir}/feature_pipelines.pkl", "wb") as f:
        pickle.dump(feature_pipelines, f)
    
    # Save calibrator
    with open(f"{model_dir}/calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    
    # Save metadata
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create model version registry entry
    registry_entry = {
        "name": model_name,
        "version": version,
        "path": model_dir,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    with open(f"{output_dir}/{model_name}/versions.json", "a+") as f:
        try:
            f.seek(0)
            content = f.read()
            if content:
                versions = json.loads(content)
            else:
                versions = []
            versions.append(registry_entry)
            f.seek(0)
            f.truncate()
            json.dump(versions, f, indent=2)
        except json.JSONDecodeError:
            # If the file is corrupted, start fresh
            f.seek(0)
            f.truncate()
            json.dump([registry_entry], f, indent=2)
```

### 2. Model Versioning

```python
def register_model_version(
    model_name: str,
    version: str,
    model_path: str,
    metadata: Dict[str, Any],
    registry_path: str
) -> None:
    """Register model version in the registry."""
    # Create registry directory if it doesn't exist
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    
    # Load existing registry or create new one
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}
    
    # Initialize model entry if it doesn't exist
    if model_name not in registry:
        registry[model_name] = {
            "versions": [],
            "current_version": None,
            "staging_version": None
        }
    
    # Add version entry
    version_entry = {
        "version": version,
        "path": model_path,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    # Check if version already exists
    for i, v in enumerate(registry[model_name]["versions"]):
        if v["version"] == version:
            # Update existing version
            registry[model_name]["versions"][i] = version_entry
            break
    else:
        # Add new version
        registry[model_name]["versions"].append(version_entry)
    
    # Set as staging version if none exists
    if registry[model_name]["staging_version"] is None:
        registry[model_name]["staging_version"] = version
    
    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
```

## Documentation & Artifact Management

The training process is documented and artifacts are managed. The following methods are used:

- **Git**: Version control for code and experiments
- **MLflow**: Tracking experiments and artifacts

## CI/CD Integration

The training process is integrated with the CI/CD pipeline. The following methods are used:

- **CI/CD**: Continuous Integration and Continuous Deployment
- **Docker**: Containerizing the training environment
- **Kubernetes**: Deploying the training environment

## Related Documentation

For more information, refer to the following documentation:

- [Feature Engineering Documentation](/docs/implementation/ml/feature_engineering.md) *(Implemented)*
- [Ad Score Prediction Implementation](/docs/implementation/ml/ad_score_prediction.md) *(Implemented)*
- [Model Evaluation](/docs/implementation/ml/model_evaluation.md) *(Planned - Not yet implemented)*
- [Ad Score Model Card](/docs/implementation/ml/model_card_ad_score_predictor.md) *(Implemented)*

> **Note**: Some of the linked documents above are currently planned but not yet implemented. Please refer to the [Documentation Tracker](/docs/implementation/documentation_tracker.md) for the current status of all documentation.

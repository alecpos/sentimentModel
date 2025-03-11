# Time Series Modeling Approach

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This technical document outlines the time series modeling approach used in WITHIN's Account Health Predictor system. It provides details on methods, implementation, challenges, and best practices for handling temporal data in advertising analytics.

## Table of Contents

- [Overview](#overview)
- [Time Series Components](#time-series-components)
- [Decomposition Methods](#decomposition-methods)
- [Forecasting Models](#forecasting-models)
- [Anomaly Detection](#anomaly-detection)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Implementation](#implementation)
- [Production Considerations](#production-considerations)
- [Best Practices](#best-practices)

## Overview

The Account Health Predictor relies heavily on time series analysis to understand temporal patterns in advertising account performance, detect anomalies, and forecast future trends. This document explains the technical approaches used to handle time-dependent data in the context of advertising account health monitoring.

Time series modeling in the WITHIN system serves several key purposes:

1. **Performance Forecasting**: Predicting future account performance metrics
2. **Trend Analysis**: Identifying long-term directional movements in key metrics
3. **Seasonality Detection**: Recognizing and accounting for cyclical patterns
4. **Anomaly Detection**: Identifying unusual patterns that may indicate account issues
5. **Causal Impact Analysis**: Measuring the impact of interventions or external factors

The system employs a multi-component approach that combines classical time series methods with modern machine learning techniques to provide robust, accurate, and interpretable results.

## Time Series Components

Time series data in advertising accounts contains multiple underlying components that need to be understood and modeled separately for effective analysis.

### Core Components

1. **Trend**: The long-term movement or direction in the data
2. **Seasonality**: Regular and predictable patterns that repeat over fixed periods
3. **Cyclical Patterns**: Irregular fluctuations without fixed frequency
4. **Noise/Residuals**: Random variation in the data

### Advertising-Specific Components

In addition to standard time series components, advertising data often exhibits:

1. **Campaign Cycles**: Patterns tied to campaign launches and conclusions
2. **Budget Exhaustion Patterns**: Regular patterns when daily or monthly budgets are consumed
3. **Day-of-Week Effects**: Consistent performance differences across days of the week
4. **Platform Algorithm Changes**: Sudden shifts due to advertising platform updates
5. **Competitive Pressure Signals**: Changes reflecting competitor activity

### Composition Models

The system employs two primary models for composing time series components:

#### Additive Model

```
Y(t) = Trend(t) + Seasonality(t) + Cyclical(t) + Residual(t)
```

Used when seasonal variations are relatively constant in magnitude over time. Common for metrics like click-through rate, conversion rate, and quality scores.

#### Multiplicative Model

```
Y(t) = Trend(t) × Seasonality(t) × Cyclical(t) × Residual(t)
```

Used when seasonal variations change proportionally with the trend. Common for metrics like impressions, clicks, costs, and conversions.

### Data Granularity Levels

The Account Health Predictor analyzes time series data at multiple granularity levels:

| Level | Granularity | Use Case | Example Metrics |
|-------|-------------|----------|----------------|
| Hourly | 1 hour | Intraday patterns | Click rate, conversion rate |
| Daily | 1 day | Performance monitoring | All core metrics |
| Weekly | 7 days | Trend analysis | Performance indices, health scores |
| Monthly | 30 days | Long-term patterns | Strategic KPIs, ROI |
| Quarterly | 90 days | Seasonal business cycles | Performance vs. goals |

## Decomposition Methods

Decomposition is a critical step in time series analysis, separating the original data into its constituent components for better understanding and modeling.

### Classical Decomposition

The system implements classical decomposition methods for baseline analysis and visualization:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_time_series(time_series, model='multiplicative', period=7):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        time_series: Pandas Series with datetime index
        model: 'additive' or 'multiplicative'
        period: Length of seasonal pattern (e.g., 7 for day-of-week)
        
    Returns:
        Object containing trend, seasonal, and residual components
    """
    result = seasonal_decompose(
        time_series,
        model=model,
        period=period,
        extrapolate_trend='freq'
    )
    return result
```

### STL Decomposition

For handling complex seasonality and robustness to outliers, Seasonal and Trend decomposition using Loess (STL) is employed:

```python
from statsmodels.tsa.seasonal import STL

def stl_decompose(time_series, period=7, robust=True):
    """
    Perform STL decomposition for complex seasonality patterns.
    
    Args:
        time_series: Pandas Series with datetime index
        period: Length of seasonal pattern
        robust: Whether to use robust fitting (less sensitive to outliers)
        
    Returns:
        STL decomposition result
    """
    stl = STL(time_series, period=period, robust=robust)
    result = stl.fit()
    return result
```

### Prophet Decomposition

For handling multiple seasonal patterns simultaneously (e.g., day-of-week, monthly, and yearly), Facebook Prophet is implemented:

```python
from prophet import Prophet

def prophet_decompose(time_series, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True):
    """
    Decompose time series using Facebook Prophet.
    
    Args:
        time_series: Pandas DataFrame with 'ds' (datetime) and 'y' (values) columns
        daily_seasonality, weekly_seasonality, yearly_seasonality: Boolean flags for seasonality components
        
    Returns:
        Prophet model and forecast components
    """
    model = Prophet(
        daily_seasonality=daily_seasonality, 
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode='multiplicative'  # Can be changed to 'additive'
    )
    
    # Add custom seasonality if needed
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(time_series)
    
    future = model.make_future_dataframe(periods=30)  # 30 days forecast
    forecast = model.predict(future)
    
    return model, forecast
```

### Wavelet Decomposition

For detecting irregular patterns and multi-scale features, wavelet decomposition is used:

```python
import pywt

def wavelet_decompose(time_series, wavelet='db8', level=3):
    """
    Perform wavelet decomposition on time series.
    
    Args:
        time_series: 1D numpy array
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Coefficients from wavelet decomposition
    """
    coeffs = pywt.wavedec(time_series, wavelet, level=level)
    return coeffs
```

### Component Analysis

After decomposition, each component is analyzed for specific characteristics:

#### Trend Analysis

```python
def analyze_trend(trend_component):
    """Analyze the trend component for direction and change points."""
    from ruptures import Pelt
    
    # Calculate slope (positive = upward trend, negative = downward trend)
    slope = np.polyfit(np.arange(len(trend_component)), trend_component, 1)[0]
    
    # Detect change points in trend
    algo = Pelt(model="rbf").fit(trend_component.values.reshape(-1, 1))
    change_points = algo.predict(pen=10)
    
    return {
        "direction": "increasing" if slope > 0 else "decreasing",
        "slope_magnitude": abs(slope),
        "change_points": change_points
    }
```

#### Seasonality Analysis

```python
def analyze_seasonality(seasonal_component, period=7):
    """Analyze the seasonality component for strength and patterns."""
    # Calculate seasonality strength
    seasonality_strength = np.std(seasonal_component) / np.std(seasonal_component + trend_component)
    
    # Find peak and trough days
    avg_by_position = np.array([
        seasonal_component[i::period].mean() 
        for i in range(period)
    ])
    
    peak_day = np.argmax(avg_by_position)
    trough_day = np.argmin(avg_by_position)
    
    return {
        "strength": seasonality_strength,
        "peak_position": peak_day,
        "trough_position": trough_day,
        "max_variation": np.max(avg_by_position) - np.min(avg_by_position)
    }
```

## Forecasting Models

The Account Health Predictor uses multiple forecasting models to predict future values of key metrics. Different models are selected based on the data characteristics, required forecast horizon, and interpretability needs.

### Statistical Models

#### ARIMA Models

Autoregressive Integrated Moving Average (ARIMA) models are used for metrics with clear autocorrelation patterns:

```python
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def fit_arima_model(time_series, exogenous=None, auto=True):
    """
    Fit ARIMA model to time series data.
    
    Args:
        time_series: Pandas Series with datetime index
        exogenous: Optional exogenous variables
        auto: Whether to use auto_arima for parameter selection
        
    Returns:
        Fitted ARIMA model
    """
    if auto:
        # Automatically find optimal parameters
        model = auto_arima(
            time_series,
            X=exogenous,
            seasonal=True,
            m=7,  # Weekly seasonality
            d=None,  # Auto-determine differencing
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            information_criterion='aic',
            error_action='ignore',
            trace=False,
            suppress_warnings=True,
            stepwise=True
        )
    else:
        # Use specified parameters (p, d, q)
        model = ARIMA(
            time_series,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 7),
            exog=exogenous
        )
        model = model.fit()
        
    return model

def forecast_arima(model, steps=30, exogenous_future=None, return_conf_int=True):
    """Generate forecast from ARIMA model."""
    forecast = model.predict(
        n_periods=steps,
        X=exogenous_future,
        return_conf_int=return_conf_int,
        alpha=0.05  # 95% confidence interval
    )
    return forecast
```

#### Exponential Smoothing

Exponential smoothing models (ETS) are used for metrics with clear trend and seasonality components:

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_ets_model(time_series, seasonal_periods=7):
    """
    Fit ETS (Exponential Smoothing) model to time series data.
    
    Args:
        time_series: Pandas Series with datetime index
        seasonal_periods: Length of seasonal cycle
        
    Returns:
        Fitted ETS model
    """
    model = ExponentialSmoothing(
        time_series,
        trend='add',             # additive trend
        seasonal='mul',          # multiplicative seasonality
        seasonal_periods=seasonal_periods,
        damped_trend=True
    )
    fitted_model = model.fit(optimized=True, use_brute=False)
    return fitted_model

def forecast_ets(model, steps=30):
    """Generate forecast from ETS model."""
    forecast = model.forecast(steps)
    return forecast
```

### Machine Learning Models

#### Prophet

Facebook Prophet is used for metrics with multiple seasonality patterns and where interpretability is important:

```python
from prophet import Prophet

def fit_prophet_model(df, changepoints=None, holidays=None, seasonality_mode='multiplicative'):
    """
    Fit Prophet model to time series data.
    
    Args:
        df: DataFrame with 'ds' (dates) and 'y' (values) columns
        changepoints: List of dates where trend changes are expected
        holidays: DataFrame of holiday events
        seasonality_mode: 'additive' or 'multiplicative'
        
    Returns:
        Fitted Prophet model
    """
    model = Prophet(
        changepoints=changepoints,
        holidays=holidays,
        seasonality_mode=seasonality_mode,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Add advertising-specific seasonality if needed
    model.add_seasonality(
        name='monthly_campaign',
        period=30.5,
        fourier_order=5
    )
    
    # Add advertising-specific regressors if available
    if 'budget' in df.columns:
        model.add_regressor('budget')
    if 'competition' in df.columns:
        model.add_regressor('competition')
        
    model.fit(df)
    return model

def forecast_prophet(model, periods=30, include_history=False):
    """Generate forecast from Prophet model."""
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    # Add future values of regressors if they were used in the model
    regressors = model.extra_regressors
    for regressor in regressors:
        # This would need to be filled with actual future values
        future[regressor['name']] = forecast_regressor(regressor['name'], periods)
        
    forecast = model.predict(future)
    
    if not include_history:
        forecast = forecast.iloc[-periods:]
        
    return forecast
```

#### LSTM Networks

Long Short-Term Memory (LSTM) networks are used for complex patterns and when high accuracy is required:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, n_steps):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, :])
        y.append(data[i+n_steps, 0])  # Target is the first column
    return np.array(X), np.array(y)

def build_lstm_model(n_steps, n_features, n_units=50, dropout=0.2):
    """Build LSTM model architecture."""
    model = Sequential()
    model.add(LSTM(units=n_units, activation='relu', return_sequences=True, 
                   input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=n_units, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_lstm_model(data, n_steps=14, n_features=None, epochs=100, batch_size=32):
    """
    Fit LSTM model to time series data.
    
    Args:
        data: Normalized numpy array with shape (n_samples, n_features)
        n_steps: Number of time steps for input sequence
        n_features: Number of features (derived from data if None)
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Fitted LSTM model and scalers
    """
    if n_features is None:
        n_features = data.shape[1]
    
    # Create sequences
    X, y = create_sequences(data, n_steps)
    
    # Split into train and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_lstm_model(n_steps, n_features)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

def forecast_lstm(model, data, n_steps, n_ahead=30, scaler=None):
    """Generate multi-step forecast from LSTM model."""
    # Initialize with last n_steps of known data
    last_sequence = data[-n_steps:].copy()
    forecast = []
    
    # Generate predictions one step at a time
    for _ in range(n_ahead):
        # Reshape for model input
        X = last_sequence.reshape(1, n_steps, last_sequence.shape[1])
        
        # Predict next value
        next_value = model.predict(X, verbose=0)[0, 0]
        forecast.append(next_value)
        
        # Update sequence with prediction for next step
        next_row = last_sequence[-1, :].copy()
        next_row[0] = next_value
        last_sequence = np.vstack([last_sequence[1:], next_row])
    
    # Inverse transform if scaler provided
    if scaler:
        forecast_reshaped = np.array(forecast).reshape(-1, 1)
        forecast_original = scaler.inverse_transform(np.hstack([
            forecast_reshaped, 
            np.zeros((len(forecast), data.shape[1]-1))
        ]))[:, 0]
        return forecast_original
    
    return np.array(forecast)
```

### Ensemble Forecasting

The system combines multiple forecasting methods to improve robustness and accuracy:

```python
def create_ensemble_forecast(time_series, forecast_horizon=30):
    """
    Create ensemble forecast combining multiple models.
    
    Args:
        time_series: Pandas Series with datetime index
        forecast_horizon: Number of periods to forecast
        
    Returns:
        DataFrame with ensemble forecast and component forecasts
    """
    # Generate forecasts from different models
    arima_forecast = forecast_arima(fit_arima_model(time_series), steps=forecast_horizon)
    ets_forecast = forecast_ets(fit_ets_model(time_series), steps=forecast_horizon)
    
    # Prepare Prophet input
    prophet_df = pd.DataFrame({
        'ds': time_series.index,
        'y': time_series.values
    })
    prophet_model = fit_prophet_model(prophet_df)
    prophet_forecast = forecast_prophet(prophet_model, periods=forecast_horizon)
    
    # Combine forecasts (simple average as baseline)
    ensemble_forecast = pd.DataFrame({
        'arima': arima_forecast,
        'ets': ets_forecast,
        'prophet': prophet_forecast['yhat'].values
    })
    
    ensemble_forecast['ensemble'] = ensemble_forecast.mean(axis=1)
    
    return ensemble_forecast
```

### Forecast Uncertainty

All forecasts include uncertainty quantification to provide confidence intervals:

```python
def generate_forecast_with_uncertainty(model, data, steps=30, method='bootstrap', samples=100):
    """
    Generate forecast with uncertainty bounds.
    
    Args:
        model: Fitted forecasting model
        data: Input time series data
        steps: Forecast horizon
        method: Method for uncertainty estimation
        samples: Number of bootstrap samples
        
    Returns:
        DataFrame with forecast mean and confidence intervals
    """
    if method == 'bootstrap':
        # Generate bootstrap samples
        bootstrap_forecasts = []
        n = len(data)
        
        for _ in range(samples):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            sample = data.iloc[indices]
            
            # Fit model on bootstrap sample
            bootstrap_model = fit_model_to_data(sample, model_type=type(model))
            
            # Generate forecast
            forecast = generate_forecast(bootstrap_model, steps)
            bootstrap_forecasts.append(forecast)
            
        # Calculate statistics
        forecasts_array = np.array(bootstrap_forecasts)
        mean_forecast = np.mean(forecasts_array, axis=0)
        lower_bound = np.percentile(forecasts_array, 2.5, axis=0)
        upper_bound = np.percentile(forecasts_array, 97.5, axis=0)
        
        return pd.DataFrame({
            'forecast': mean_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    elif method == 'analytical':
        # Use model's analytical confidence intervals if available
        # Implementation depends on specific model type
        pass
```

## Anomaly Detection

Anomaly detection is a critical component of the Account Health Predictor, identifying unusual patterns that may indicate account issues or opportunities. Multiple complementary techniques are employed for robust detection.

### Statistical Methods

#### Z-Score Method

Used for detecting point anomalies based on deviation from the mean:

```python
def z_score_anomalies(time_series, threshold=3.0, window=None):
    """
    Detect anomalies using Z-score method.
    
    Args:
        time_series: Pandas Series with datetime index
        threshold: Z-score threshold for anomaly detection
        window: Rolling window size (None = use entire series)
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    if window is None:
        # Global mean and std
        mean = time_series.mean()
        std = time_series.std()
        z_scores = (time_series - mean) / std
    else:
        # Rolling mean and std
        mean = time_series.rolling(window=window, center=True).mean()
        std = time_series.rolling(window=window, center=True).std()
        z_scores = (time_series - mean) / std
        
    return abs(z_scores) > threshold
```

#### Moving Average Method

Detecting anomalies based on deviation from expected values:

```python
def moving_average_anomalies(time_series, window=7, threshold=2.0):
    """
    Detect anomalies using moving average method.
    
    Args:
        time_series: Pandas Series with datetime index
        window: Size of moving average window
        threshold: Number of standard deviations for anomaly threshold
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    # Calculate moving average and standard deviation
    moving_avg = time_series.rolling(window=window, center=True).mean()
    moving_std = time_series.rolling(window=window, center=True).std()
    
    # Calculate lower and upper bounds
    lower_bound = moving_avg - (threshold * moving_std)
    upper_bound = moving_avg + (threshold * moving_std)
    
    # Identify anomalies
    anomalies = (time_series < lower_bound) | (time_series > upper_bound)
    
    return anomalies
```

#### Seasonal Decomposition Method

Detecting anomalies based on residual component after decomposition:

```python
def seasonal_decomposition_anomalies(time_series, period=7, threshold=2.5, model='multiplicative'):
    """
    Detect anomalies using seasonal decomposition residuals.
    
    Args:
        time_series: Pandas Series with datetime index
        period: Seasonal period
        threshold: Number of standard deviations for anomaly threshold
        model: Decomposition model ('additive' or 'multiplicative')
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(time_series, model=model, period=period)
    residuals = decomposition.resid
    
    # Calculate mean and standard deviation of residuals
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    
    # Identify anomalies
    anomalies = abs(residuals - residual_mean) > (threshold * residual_std)
    
    return anomalies
```

### Machine Learning Methods

#### Isolation Forest

Used for unsupervised anomaly detection based on isolation of observations:

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_anomalies(time_series, contamination=0.05, features=None):
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Args:
        time_series: Pandas Series with datetime index
        contamination: Expected proportion of anomalies
        features: Additional features to include in detection
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    # Prepare data
    if features is None:
        # Use time series values only
        X = time_series.values.reshape(-1, 1)
    else:
        # Combine time series with additional features
        X = np.hstack([
            time_series.values.reshape(-1, 1),
            features.values
        ])
    
    # Fit Isolation Forest model
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    model.fit(X)
    
    # Predict anomalies
    # -1 for anomalies, 1 for normal values
    predictions = model.predict(X)
    
    # Convert to boolean Series
    anomalies = pd.Series(predictions == -1, index=time_series.index)
    
    return anomalies
```

#### DBSCAN

Used for density-based anomaly detection:

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan_anomalies(time_series, eps=0.5, min_samples=5, features=None):
    """
    Detect anomalies using DBSCAN clustering algorithm.
    
    Args:
        time_series: Pandas Series with datetime index
        eps: Maximum distance between samples for clustering
        min_samples: Minimum samples in a cluster
        features: Additional features to include in detection
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    # Prepare data
    if features is None:
        # Use time series values only
        X = time_series.values.reshape(-1, 1)
    else:
        # Combine time series with additional features
        X = np.hstack([
            time_series.values.reshape(-1, 1),
            features.values
        ])
    
    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Fit DBSCAN model
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)
    
    # Points labeled as -1 are anomalies (noise)
    anomalies = pd.Series(clusters == -1, index=time_series.index)
    
    return anomalies
```

#### LSTM Autoencoder

Used for detecting complex temporal anomalies:

```python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

def create_lstm_autoencoder(seq_length, n_features):
    """Create LSTM autoencoder for anomaly detection."""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, n_features), return_sequences=False),
        RepeatVector(seq_length),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def lstm_autoencoder_anomalies(time_series, seq_length=10, threshold_multiplier=3.0, features=None):
    """
    Detect anomalies using LSTM autoencoder reconstruction error.
    
    Args:
        time_series: Pandas Series with datetime index
        seq_length: Sequence length for LSTM
        threshold_multiplier: Multiplier for reconstruction error threshold
        features: Additional features to include
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    # Prepare data
    if features is None:
        # Use time series values only
        data = time_series.values.reshape(-1, 1)
        n_features = 1
    else:
        # Combine time series with additional features
        data = np.hstack([
            time_series.values.reshape(-1, 1),
            features.values
        ])
        n_features = data.shape[1]
    
    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X = np.array([data_scaled[i:i+seq_length] for i in range(len(data_scaled)-seq_length)])
    
    # Create and train autoencoder
    model = create_lstm_autoencoder(seq_length, n_features)
    model.fit(X, X, epochs=50, batch_size=32, verbose=0)
    
    # Calculate reconstruction error
    X_pred = model.predict(X)
    mse = np.mean(np.square(X - X_pred), axis=(1, 2))
    
    # Set threshold based on reconstruction error distribution
    threshold = np.mean(mse) + threshold_multiplier * np.std(mse)
    
    # Identify anomalies
    anomaly_indices = np.where(mse > threshold)[0]
    
    # Convert to full time series
    anomalies = np.zeros(len(time_series), dtype=bool)
    
    # Mark anomalous sequences in the original time series
    for idx in anomaly_indices:
        anomalies[idx:idx+seq_length] = True
    
    return pd.Series(anomalies, index=time_series.index)
```

### Ensemble Anomaly Detection

Combining multiple detection methods for robust results:

```python
def ensemble_anomaly_detection(time_series, methods=None, min_detections=2):
    """
    Combine multiple anomaly detection methods.
    
    Args:
        time_series: Pandas Series with datetime index
        methods: List of anomaly detection methods to use
        min_detections: Minimum number of methods that must detect an anomaly
        
    Returns:
        Series of boolean values (True for anomalies)
    """
    if methods is None:
        methods = [
            z_score_anomalies,
            moving_average_anomalies,
            seasonal_decomposition_anomalies,
            isolation_forest_anomalies
        ]
    
    # Apply each detection method
    all_anomalies = []
    for method in methods:
        try:
            anomalies = method(time_series)
            all_anomalies.append(anomalies)
        except Exception as e:
            print(f"Error in method {method.__name__}: {e}")
            continue
    
    # Count how many methods detected each point as anomaly
    anomaly_counts = sum(all_anomalies)
    
    # Final anomalies are those detected by at least min_detections methods
    final_anomalies = anomaly_counts >= min_detections
    
    return final_anomalies

def get_anomaly_details(time_series, anomalies, window=3):
    """
    Get details for detected anomalies.
    
    Args:
        time_series: Pandas Series with datetime index
        anomalies: Boolean Series marking anomalies
        window: Context window size
        
    Returns:
        DataFrame with anomaly details
    """
    anomaly_indices = np.where(anomalies)[0]
    
    results = []
    for idx in anomaly_indices:
        # Get timestamp
        timestamp = time_series.index[idx]
        
        # Get value
        value = time_series.iloc[idx]
        
        # Get context window
        start_idx = max(0, idx - window)
        end_idx = min(len(time_series), idx + window + 1)
        context = time_series.iloc[start_idx:end_idx]
        
        # Calculate deviation from expected
        expected = context.mean()
        deviation = (value - expected) / expected if expected != 0 else float('inf')
        
        results.append({
            'timestamp': timestamp,
            'value': value,
            'expected': expected,
            'deviation': deviation,
            'deviation_pct': deviation * 100
        })
    
    return pd.DataFrame(results)
```

## Feature Engineering

Feature engineering is crucial for extracting valuable signals from raw time series data. The Account Health Predictor employs various feature engineering techniques to maximize predictive power.

### Temporal Features

#### Basic Calendar Features

```python
def add_calendar_features(df, date_column):
    """
    Add calendar-based features to time series data.
    
    Args:
        df: Pandas DataFrame
        date_column: Name of date column
        
    Returns:
        DataFrame with additional calendar features
    """
    # Ensure date column is datetime
    df = df.copy()
    if df[date_column].dtype != 'datetime64[ns]':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract basic calendar features
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['day_of_month'] = df[date_column].dt.day
    df['month'] = df[date_column].dt.month
    df['quarter'] = df[date_column].dt.quarter
    df['year'] = df[date_column].dt.year
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    
    # Add cyclic encoding for cyclical features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df
```

#### Holiday and Event Features

```python
from holidays import US

def add_holiday_features(df, date_column, country='US'):
    """
    Add holiday-related features to time series data.
    
    Args:
        df: Pandas DataFrame
        date_column: Name of date column
        country: Country for holidays (default: US)
        
    Returns:
        DataFrame with holiday features
    """
    df = df.copy()
    
    # Initialize holiday calendar
    if country == 'US':
        holiday_calendar = US()
    # Add other countries as needed
    
    # Mark holidays
    df['is_holiday'] = df[date_column].apply(
        lambda x: x in holiday_calendar
    ).astype(int)
    
    # Mark days before and after holidays
    holiday_dates = df[df['is_holiday'] == 1][date_column].tolist()
    df['day_before_holiday'] = df[date_column].apply(
        lambda x: (x + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df['day_after_holiday'] = df[date_column].apply(
        lambda x: (x - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    
    # Add specific major holiday flags
    major_holidays = [
        'New Year\'s Day',
        'Independence Day',
        'Thanksgiving',
        'Christmas Day',
        'Memorial Day',
        'Labor Day'
    ]
    
    for holiday in major_holidays:
        holiday_dates = [
            date for date, name in holiday_calendar.items()
            if holiday in name
        ]
        df[f'is_{holiday.lower().replace("\'", "").replace(" ", "_")}'] = df[date_column].apply(
            lambda x: x in holiday_dates
        ).astype(int)
    
    # Add retail season flags
    df['is_black_friday_period'] = (
        (df[date_column].dt.month == 11) & 
        (df[date_column].dt.day >= 20) & 
        (df[date_column].dt.day <= 30)
    ).astype(int)
    
    df['is_cyber_monday_period'] = (
        (df[date_column].dt.month == 11) & 
        (df[date_column].dt.day >= 25)
    ) | (
        (df[date_column].dt.month == 12) & 
        (df[date_column].dt.day <= 5)
    ).astype(int)
    
    df['is_holiday_shopping_season'] = (
        (df[date_column].dt.month == 11) & 
        (df[date_column].dt.day >= 20)
    ) | (
        (df[date_column].dt.month == 12) & 
        (df[date_column].dt.day <= 24)
    ).astype(int)
    
    return df
```

#### Advertising-Specific Temporal Features

```python
def add_ad_specific_temporal_features(df, date_column):
    """
    Add advertising-specific temporal features.
    
    Args:
        df: Pandas DataFrame
        date_column: Name of date column
        
    Returns:
        DataFrame with advertising-specific features
    """
    df = df.copy()
    
    # End of month flag (common for budget exhaustion)
    df['is_end_of_month'] = (
        df[date_column].dt.day >= 25
    ).astype(int)
    
    # Start of month flag (common for budget refreshes)
    df['is_start_of_month'] = (
        df[date_column].dt.day <= 5
    ).astype(int)
    
    # Time of quarter features
    df['days_from_quarter_start'] = df[date_column].dt.dayofyear - (
        df[date_column].dt.quarter - 1) * 91
    df['is_end_of_quarter'] = (
        df['days_from_quarter_start'] >= 80
    ).astype(int)
    
    # Day of week advertising performance patterns
    # Monday: 0, Sunday: 6
    df['is_high_performance_day'] = df['day_of_week'].isin([1, 2, 3]).astype(int)  # Tue-Thu
    df['is_low_performance_day'] = df['day_of_week'].isin([5, 6]).astype(int)  # Sat-Sun
    
    # Monthly seasonality (some industries have monthly patterns)
    df['week_of_month'] = (df[date_column].dt.day - 1) // 7 + 1
    
    return df
```

### Statistical Features

#### Lagged Features

```python
def add_lagged_features(df, target_column, lags=None):
    """
    Add lagged values of target variable as features.
    
    Args:
        df: Pandas DataFrame
        target_column: Column to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    df = df.copy()
    
    if lags is None:
        lags = [1, 7, 14, 28]  # Default lags for daily data
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    return df
```

#### Rolling Window Features

```python
def add_rolling_features(df, target_column, windows=None, functions=None):
    """
    Add rolling window statistics as features.
    
    Args:
        df: Pandas DataFrame
        target_column: Column to compute statistics for
        windows: List of window sizes
        functions: List of functions to apply
        
    Returns:
        DataFrame with rolling window features
    """
    df = df.copy()
    
    if windows is None:
        windows = [7, 14, 30]  # Default windows for daily data
    
    if functions is None:
        functions = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'median': np.median
        }
    
    for window in windows:
        for name, func in functions.items():
            df[f'{target_column}_{name}_{window}d'] = (
                df[target_column].rolling(window=window, min_periods=1).apply(func)
            )
    
    return df
```

#### Ratio Features

```python
def add_ratio_features(df, target_column):
    """
    Add ratio features comparing recent values to historical values.
    
    Args:
        df: Pandas DataFrame
        target_column: Target column
        
    Returns:
        DataFrame with ratio features
    """
    df = df.copy()
    
    # Ratio to previous day
    df[f'{target_column}_ratio_1d'] = df[target_column] / df[f'{target_column}_lag_1']
    
    # Ratio to previous week
    df[f'{target_column}_ratio_7d'] = df[target_column] / df[f'{target_column}_lag_7']
    
    # Ratio to previous short-term average
    df[f'{target_column}_ratio_to_7d_avg'] = (
        df[target_column] / df[f'{target_column}_mean_7d']
    )
    
    # Ratio to previous longer-term average
    df[f'{target_column}_ratio_to_30d_avg'] = (
        df[target_column] / df[f'{target_column}_mean_30d']
    )
    
    # Compare current week to previous week
    df[f'{target_column}_week_over_week'] = (
        df[f'{target_column}_mean_7d'] / df[f'{target_column}_mean_14d']
    )
    
    # Volatility ratios
    df[f'{target_column}_volatility_ratio'] = (
        df[f'{target_column}_std_7d'] / df[f'{target_column}_std_30d']
    )
    
    return df
```

### Trend and Seasonality Features

#### Trend Indicators

```python
def add_trend_features(df, target_column, windows=None):
    """
    Add trend-related features.
    
    Args:
        df: Pandas DataFrame
        target_column: Target column
        windows: Windows for trend computation
        
    Returns:
        DataFrame with trend features
    """
    df = df.copy()
    
    if windows is None:
        windows = [7, 14, 30]
    
    for window in windows:
        # Linear trend coefficient for window
        df[f'{target_column}_trend_{window}d'] = np.nan
        
        for i in range(window, len(df)):
            y = df[target_column].iloc[i-window:i].values
            x = np.arange(window)
            slope, _ = np.polyfit(x, y, 1)
            df[f'{target_column}_trend_{window}d'].iloc[i] = slope
        
        # Trend direction (1: up, 0: flat, -1: down)
        threshold = 0.001
        df[f'{target_column}_trend_direction_{window}d'] = np.where(
            df[f'{target_column}_trend_{window}d'] > threshold, 1,
            np.where(df[f'{target_column}_trend_{window}d'] < -threshold, -1, 0)
        )
    
    return df
```

#### Fourier Features for Seasonality

```python
def add_fourier_features(df, date_column, periods=None, orders=None):
    """
    Add Fourier terms for capturing seasonality.
    
    Args:
        df: Pandas DataFrame
        date_column: Date column name
        periods: List of seasonal periods in days
        orders: Number of fourier terms for each period
        
    Returns:
        DataFrame with Fourier features
    """
    df = df.copy()
    
    if periods is None:
        periods = [7, 30.5, 365.25]  # Weekly, monthly, yearly seasonality
    
    if orders is None:
        orders = [3, 5, 10]  # Higher order for longer periods
    
    # Convert datetime to ordinal days for calculation
    df['_ordinal_day'] = pd.to_datetime(df[date_column]).map(lambda x: x.toordinal())
    
    for period, order in zip(periods, orders):
        for n in range(1, order + 1):
            # Sine component
            df[f'seasonal_sin_{int(period)}d_{n}'] = np.sin(
                2 * np.pi * n * df['_ordinal_day'] / period
            )
            
            # Cosine component
            df[f'seasonal_cos_{int(period)}d_{n}'] = np.cos(
                2 * np.pi * n * df['_ordinal_day'] / period
            )
    
    # Drop helper column
    df = df.drop('_ordinal_day', axis=1)
    
    return df
```

### External and Domain-Specific Features

#### Ad Platform Features

```python
def add_platform_specific_features(df, platform_column=None):
    """
    Add advertising platform-specific features.
    
    Args:
        df: Pandas DataFrame
        platform_column: Column containing platform names
        
    Returns:
        DataFrame with platform-specific features
    """
    df = df.copy()
    
    if platform_column:
        # Create platform dummy variables
        platforms = df[platform_column].unique()
        for platform in platforms:
            df[f'is_{platform}'] = (df[platform_column] == platform).astype(int)
    
    # Add platform-specific marketing calendar events
    # These would typically come from a separate calendar data source
    # This is a placeholder implementation
    platform_events = {
        'black_friday_sale': ['2023-11-24', '2023-11-27'],
        'holiday_sale': ['2023-12-10', '2023-12-25'],
        'summer_sale': ['2023-07-01', '2023-07-15']
    }
    
    for event_name, dates in platform_events.items():
        date_range = pd.date_range(start=dates[0], end=dates[1])
        df[f'is_{event_name}'] = df[date_column].isin(date_range).astype(int)
    
    return df
```

#### Industry-Specific Features

```python
def add_industry_features(df, industry_column=None, date_column=None):
    """
    Add industry-specific features.
    
    Args:
        df: Pandas DataFrame
        industry_column: Column with industry information
        date_column: Date column
        
    Returns:
        DataFrame with industry-specific features
    """
    df = df.copy()
    
    if industry_column and date_column:
        industries = df[industry_column].unique()
        
        # Example industry-specific seasonal patterns
        for industry in industries:
            if industry == 'retail':
                # Retail-specific seasonal flags
                df.loc[df[industry_column] == 'retail', 'is_peak_season'] = (
                    (df[date_column].dt.month >= 11) | 
                    (df[date_column].dt.month <= 1)
                ).astype(int)
                
            elif industry == 'travel':
                # Travel-specific seasonal flags
                df.loc[df[industry_column] == 'travel', 'is_peak_season'] = (
                    (df[date_column].dt.month >= 6) & 
                    (df[date_column].dt.month <= 8)
                ).astype(int)
    
    return df
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def select_best_features(X, y, method='f_regression', k=10):
    """
    Select most important features for time series forecasting.
    
    Args:
        X: Feature DataFrame
        y: Target series
        method: Selection method ('f_regression' or 'mutual_info')
        k: Number of features to select
        
    Returns:
        Selected feature names
    """
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit selector
    selector.fit(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices].tolist()
    
    # Get feature scores
    scores = selector.scores_
    feature_scores = dict(zip(X.columns, scores))
    
    sorted_features = sorted(
        feature_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return selected_features, sorted_features
```

## Model Evaluation

Evaluating time series models requires specialized techniques that account for temporal dependence and forecasting accuracy across different time horizons.

### Time Series Cross-Validation

```python
def time_series_cv(model_func, data, target_column, n_splits=5, train_size=0.7, test_window=30):
    """
    Perform time series cross-validation.
    
    Args:
        model_func: Function that builds and fits a model given train data
        data: DataFrame with time series data
        target_column: Target column name
        n_splits: Number of validation splits
        train_size: Initial training size as fraction of data
        test_window: Number of periods in each test window
        
    Returns:
        Dictionary with performance metrics for each split
    """
    # Ensure data is sorted by time
    data = data.sort_index()
    
    n_samples = len(data)
    initial_train_size = int(n_samples * train_size)
    
    results = []
    
    for i in range(n_splits):
        # Calculate split indices
        train_end = initial_train_size + i * test_window
        test_start = train_end
        test_end = min(test_start + test_window, n_samples)
        
        # Skip if we've reached the end of data
        if test_start >= n_samples:
            break
        
        # Create train/test splits
        train_data = data.iloc[:train_end]
        test_data = data.iloc[test_start:test_end]
        
        # Fit model and generate predictions
        model = model_func(train_data)
        predictions = model.predict(test_data.drop(target_column, axis=1))
        
        # Calculate metrics
        actual = test_data[target_column].values
        metrics = calculate_forecast_metrics(actual, predictions)
        metrics['split'] = i + 1
        metrics['train_size'] = len(train_data)
        metrics['test_size'] = len(test_data)
        
        results.append(metrics)
    
    return pd.DataFrame(results)
```

### Forecast Error Metrics

```python
def calculate_forecast_metrics(actual, predicted, horizon=None):
    """
    Calculate error metrics for forecasting.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        horizon: Optional forecast horizon
        
    Returns:
        Dictionary of error metrics
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Handle horizons if provided
    if horizon is not None:
        # Calculate metrics per horizon
        horizons = range(1, horizon + 1)
        metrics_by_horizon = []
        
        for h in horizons:
            actual_h = actual[h-1::horizon]
            predicted_h = predicted[h-1::horizon]
            
            if len(actual_h) > 0:
                metrics = {
                    'horizon': h,
                    'mse': mean_squared_error(actual_h, predicted_h),
                    'rmse': np.sqrt(mean_squared_error(actual_h, predicted_h)),
                    'mae': mean_absolute_error(actual_h, predicted_h),
                    'mape': np.mean(np.abs((actual_h - predicted_h) / (actual_h + 1e-8))) * 100,
                    'r2': r2_score(actual_h, predicted_h),
                    'samples': len(actual_h)
                }
                metrics_by_horizon.append(metrics)
        
        return metrics_by_horizon
    
    # Calculate overall metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Mean Absolute Percentage Error (handle zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    # Calculate R-squared
    r2 = r2_score(actual, predicted)
    
    # Direction accuracy (correct trend direction prediction)
    actual_diff = np.diff(actual)
    predicted_diff = np.diff(predicted)
    direction_match = np.sign(actual_diff) == np.sign(predicted_diff)
    direction_accuracy = np.mean(direction_match) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual) + 1e-8)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }
```

### Visualization and Diagnostics

```python
def plot_forecast_vs_actual(actual, predicted, title='Forecast vs Actual'):
    """
    Plot predicted values against actual values.
    
    Args:
        actual: Series of actual values with datetime index
        predicted: Series of predicted values with datetime index
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(predicted.index, predicted, label='Forecast', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add error regions
    error = np.abs(actual - predicted)
    plt.fill_between(
        actual.index,
        predicted - error, 
        predicted + error,
        color='red', 
        alpha=0.2, 
        label='Error Region'
    )
    
    plt.tight_layout()
    plt.show()

def plot_residual_diagnostics(actual, predicted):
    """
    Plot residual diagnostics for forecast evaluation.
    
    Args:
        actual: Series of actual values
        predicted: Series of predicted values
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(actual.index, residuals, color='blue')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, color='blue', alpha=0.7)
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs. Predicted
    axes[1, 1].scatter(predicted, residuals, color='blue', alpha=0.5)
    axes[1, 1].set_title('Residuals vs. Predicted Values')
    axes[1, 1].axhline(y=0, color='r', linestyle='-')
    axes[1, 1].set_xlabel('Predicted Value')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(time_series, lags=40):
    """
    Plot Autocorrelation and Partial Autocorrelation functions.
    
    Args:
        time_series: Time series data
        lags: Number of lags to include
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    plot_acf(time_series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function')
    
    # PACF
    plot_pacf(time_series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()
```

## Implementation

### Full Time Series Processing Pipeline

The complete time series pipeline for the Account Health Predictor integrates data fetching, preprocessing, feature engineering, modeling, and result processing:

```python
class TimeSeriesPipeline:
    """End-to-end pipeline for time series processing."""
    
    def __init__(self, config=None):
        """Initialize the pipeline with configuration."""
        self.config = config or {}
        self.models = {}
        self.preprocessors = {}
        self.metrics = {}
    
    def fetch_data(self, account_id, platform, start_date, end_date):
        """Fetch raw time series data for the account."""
        # Implementation depends on data source
        # This is a placeholder
        from within.data.connectors import DataConnector
        
        connector = DataConnector()
        raw_data = connector.get_account_data(
            account_id=account_id,
            platform=platform,
            start_date=start_date,
            end_date=end_date
        )
        
        return raw_data
    
    def preprocess_data(self, raw_data):
        """Preprocess raw data for modeling."""
        # Basic preprocessing steps
        df = raw_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Normalize/standardize features
        df = self._normalize_features(df)
        
        return df
    
    def engineer_features(self, df):
        """Apply feature engineering to preprocessed data."""
        # Add various feature types
        df = add_calendar_features(df, 'date')
        df = add_lagged_features(df, 'metric_value', lags=[1, 7, 14, 30])
        df = add_rolling_features(df, 'metric_value')
        df = add_ratio_features(df, 'metric_value')
        df = add_trend_features(df, 'metric_value')
        df = add_fourier_features(df, 'date')
        
        return df
    
    def build_models(self, training_data, target_column, features):
        """Build and train time series models."""
        # Initialize models
        from within.models.time_series import (
            ARIMAModel, ProphetModel, LSTMModel, 
            EnsembleTimeSeriesModel
        )
        
        models = {
            'arima': ARIMAModel(),
            'prophet': ProphetModel(),
            'lstm': LSTMModel(),
        }
        
        # Train each model
        for name, model in models.items():
            model.fit(
                training_data,
                target_column=target_column,
                features=features
            )
            self.models[name] = model
        
        # Create ensemble model
        ensemble = EnsembleTimeSeriesModel(models=list(models.values()))
        ensemble.fit(training_data, target_column=target_column, features=features)
        self.models['ensemble'] = ensemble
        
        return self.models
    
    def detect_anomalies(self, data, target_column):
        """Detect anomalies in time series data."""
        # Apply multiple anomaly detection methods
        anomalies = ensemble_anomaly_detection(
            data[target_column],
            methods=[
                z_score_anomalies,
                moving_average_anomalies,
                isolation_forest_anomalies
            ],
            min_detections=2
        )
        
        return anomalies
    
    def generate_forecasts(self, horizon=30):
        """Generate forecasts using trained models."""
        forecasts = {}
        
        # Generate forecast from each model
        for name, model in self.models.items():
            forecast = model.predict(steps=horizon)
            forecasts[name] = forecast
        
        return forecasts
    
    def evaluate_forecasts(self, actual, forecasts):
        """Evaluate forecast accuracy."""
        metrics = {}
        
        # Calculate metrics for each model
        for name, forecast in forecasts.items():
            metrics[name] = calculate_forecast_metrics(actual, forecast)
        
        self.metrics = metrics
        return metrics
    
    def run_pipeline(self, account_id, platform, start_date, end_date, forecast_horizon=30):
        """Run the complete pipeline."""
        # Fetch data
        raw_data = self.fetch_data(account_id, platform, start_date, end_date)
        
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Engineer features
        featured_data = self.engineer_features(processed_data)
        
        # Split data into train/test
        train_data, test_data = self._train_test_split(featured_data)
        
        # Build and train models
        target_column = 'metric_value'
        feature_columns = [col for col in featured_data.columns if col != target_column]
        self.build_models(train_data, target_column, feature_columns)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(featured_data, target_column)
        
        # Generate forecasts
        forecasts = self.generate_forecasts(horizon=forecast_horizon)
        
        # Evaluate forecasts
        if test_data is not None:
            metrics = self.evaluate_forecasts(
                test_data[target_column].values,
                {name: f[:len(test_data)] for name, f in forecasts.items()}
            )
        
        return {
            'processed_data': processed_data,
            'featured_data': featured_data,
            'anomalies': anomalies,
            'forecasts': forecasts,
            'metrics': self.metrics
        }
    
    def _handle_missing_values(self, df):
        """Handle missing values in time series data."""
        # First, check for gaps in date sequence
        date_column = 'date'
        full_date_range = pd.date_range(
            start=df[date_column].min(),
            end=df[date_column].max(),
            freq='D'
        )
        
        # Reindex to include all dates
        df = df.set_index(date_column).reindex(full_date_range).reset_index()
        df = df.rename(columns={'index': date_column})
        
        # Forward fill for short gaps (up to 3 days)
        df = df.fillna(method='ffill', limit=3)
        
        # Use rolling mean for remaining gaps
        for column in df.columns:
            if column != date_column and df[column].isna().any():
                df[column] = df[column].fillna(
                    df[column].rolling(window=14, min_periods=1, center=True).mean()
                )
        
        return df
    
    def _convert_data_types(self, df):
        """Convert columns to appropriate data types."""
        # Ensure date column is datetime
        date_column = 'date'
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Convert numeric columns
        numeric_columns = ['metric_value', 'impressions', 'clicks', 'conversions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _normalize_features(self, df):
        """Normalize features for modeling."""
        from sklearn.preprocessing import StandardScaler, RobustScaler
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Create and fit scaler
        scaler = RobustScaler()  # More robust to outliers
        
        # Store scaler for later use
        self.preprocessors['scaler'] = scaler
        
        # Scale numeric features
        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def _train_test_split(self, df, test_size=0.2):
        """Split time series data respecting temporal order."""
        n = len(df)
        train_size = int(n * (1 - test_size))
        
        train = df.iloc[:train_size].copy()
        test = df.iloc[train_size:].copy()
        
        return train, test
```

## Production Considerations

Deploying time series models in production requires attention to several technical considerations to ensure efficient operation, maintainability, and accuracy.

### Scheduling and Updating

Time series models require regular updating as new data becomes available:

```python
def schedule_model_updates(account_ids, frequency='daily'):
    """
    Schedule regular updates of time series models.
    
    Args:
        account_ids: List of accounts to update
        frequency: Update frequency
    """
    from within.scheduling import Scheduler
    
    scheduler = Scheduler()
    
    if frequency == 'daily':
        cron_expression = '0 3 * * *'  # 3 AM daily
    elif frequency == 'weekly':
        cron_expression = '0 3 * * 1'  # 3 AM on Mondays
    elif frequency == 'monthly':
        cron_expression = '0 3 1 * *'  # 3 AM on 1st of month
    
    # Register update job
    job_id = scheduler.register_job(
        function=update_account_models,
        args=(account_ids,),
        cron_expression=cron_expression,
        job_name='time_series_model_update',
        description='Regular update of account health time series models'
    )
    
    scheduler.start_job(job_id)
    
    return job_id

def update_account_models(account_ids):
    """Update models for specified accounts."""
    from within.logging import get_logger
    logger = get_logger('model_updates')
    
    for account_id in account_ids:
        try:
            # Initialize pipeline
            pipeline = TimeSeriesPipeline()
            
            # Get latest data (last 90 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            
            # Run pipeline
            result = pipeline.run_pipeline(
                account_id=account_id,
                platform='all',
                start_date=start_date,
                end_date=end_date,
                forecast_horizon=30
            )
            
            # Save updated models
            save_models(account_id, pipeline.models)
            
            logger.info(f"Successfully updated models for account {account_id}")
            
        except Exception as e:
            logger.error(f"Error updating models for account {account_id}: {str(e)}")
```

### Caching and Performance Optimization

Optimize performance by implementing caching for time series computations:

```python
def implement_forecast_caching(cache_duration=3600):
    """
    Implement caching for forecasts to improve API performance.
    
    Args:
        cache_duration: Cache lifetime in seconds
    """
    from within.caching import Cache
    
    # Initialize cache
    forecast_cache = Cache(
        name='account_forecasts',
        ttl=cache_duration
    )
    
    # Example usage in API endpoint
    def get_account_forecast_api(account_id, platform, horizon=30):
        """API endpoint for account forecasts with caching."""
        # Cache key
        cache_key = f"forecast:{account_id}:{platform}:{horizon}"
        
        # Try to get from cache
        cached_forecast = forecast_cache.get(cache_key)
        if cached_forecast is not None:
            return cached_forecast
        
        # Not in cache, generate forecast
        pipeline = TimeSeriesPipeline()
        forecast = pipeline.run_pipeline(
            account_id=account_id,
            platform=platform,
            start_date=datetime.now().date() - timedelta(days=90),
            end_date=datetime.now().date(),
            forecast_horizon=horizon
        )['forecasts']['ensemble']
        
        # Store in cache
        forecast_cache.set(cache_key, forecast)
        
        return forecast
```

### Monitoring and Alerting

Implement monitoring for forecast accuracy and anomalies:

```python
def setup_forecast_monitoring(threshold=0.15):
    """
    Set up monitoring for forecast accuracy.
    
    Args:
        threshold: MAPE threshold for alerts
    """
    from within.monitoring import Monitor
    
    monitor = Monitor(name='forecast_accuracy')
    
    # Register metrics to track
    monitor.register_metric(
        name='forecast_mape',
        description='Mean Absolute Percentage Error of forecasts',
        unit='percent',
        threshold=threshold,
        alert_channels=['slack-ml-team', 'email-alerts']
    )
    
    # Register callback for monitoring
    def track_forecast_accuracy(actual, predicted, metadata):
        """Calculate and track forecast accuracy."""
        metrics = calculate_forecast_metrics(actual, predicted)
        
        monitor.log_metric(
            name='forecast_mape',
            value=metrics['mape'],
            metadata=metadata
        )
        
        # Log additional metrics without alerting
        monitor.log_metric(
            name='forecast_rmse',
            value=metrics['rmse'],
            metadata=metadata,
            alert=False
        )
    
    return track_forecast_accuracy
```

## Best Practices

### Time Series Data Handling

1. **Consistent Frequency**: Ensure time series data has consistent frequency and reindex to fill gaps
2. **Outlier Handling**: Use robust methods (median, IQR) for outlier detection and treatment 
3. **Stationarity**: Test for and address non-stationarity through differencing or transformations
4. **Change Point Detection**: Identify structural breaks in time series and adjust models accordingly
5. **Domain Knowledge**: Incorporate domain knowledge about advertising seasonality and platform changes

### Modeling Approaches

1. **Start Simple**: Begin with simple models (ARIMA, ETS) before moving to complex ones (LSTM, deep learning)
2. **Ensemble Methods**: Combine multiple forecasting methods for robustness
3. **Regular Evaluation**: Continuously evaluate model performance as new data arrives
4. **Horizon-Specific Models**: Consider different models for different forecast horizons
5. **Feature Importance**: Analyze feature importance to understand driving factors

### Implementation Tips

1. **Reproducibility**: Use fixed random seeds for probabilistic components
2. **Computational Efficiency**: Optimize heavy computations and use caching where appropriate
3. **Logging**: Log all model training, forecasts, and anomalies for audit and analysis
4. **Modularity**: Design components to be independently testable and replaceable
5. **Meta-Features**: Track meta-features about account data to inform model selection

### Production Deployment

1. **Versioning**: Version all models and make version transitions seamless
2. **Gradual Rollout**: Roll out new time series models gradually to detect issues early
3. **Shadow Deployment**: Run new models in shadow mode alongside existing ones
4. **Monitoring**: Implement comprehensive monitoring of model performance
5. **Fallback Mechanisms**: Have fallback strategies for when models underperform

By following these guidelines and implementing the approaches detailed in this document, the Account Health Predictor system can effectively leverage time series modeling techniques to deliver accurate, timely, and actionable insights for advertising account health management. 
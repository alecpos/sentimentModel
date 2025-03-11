# Troubleshooting Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides solutions to common issues you might encounter when using the WITHIN Ad Score & Account Health Predictor system. It covers API integration, model performance, data issues, and system access problems.

## Table of Contents

- [API Issues](#api-issues)
- [Model Performance Issues](#model-performance-issues)
- [Data Integration Issues](#data-integration-issues)
- [Authentication Problems](#authentication-problems)
- [Dashboard Problems](#dashboard-problems)
- [Performance Bottlenecks](#performance-bottlenecks)
- [Common Error Codes](#common-error-codes)
- [Logging and Debugging](#logging-and-debugging)
- [Support Resources](#support-resources)

This guide is designed to help you resolve common issues quickly. For more specialized problems, contact our support team at support@within.co.

## API Issues

### Connection Failures

**Problem**: API requests fail with connection errors.

**Troubleshooting Steps**:
1. Verify the API endpoint URL is correct
2. Check your network connection and firewalls
3. Ensure your DNS is resolving correctly
4. Try using a different network to rule out local network issues

**Solution Example**:
```python
# Check API connectivity
import requests
try:
    response = requests.get("https://api.within.co/api/v1/health")
    print(f"Connection successful, status code: {response.status_code}")
except requests.exceptions.ConnectionError as e:
    print(f"Connection failed: {e}")
```

### Authentication Errors

**Problem**: Getting 401 Unauthorized or 403 Forbidden responses.

**Troubleshooting Steps**:
1. Verify your API keys are correct and active
2. Check that the signature is being correctly generated
3. Ensure your system clock is synchronized (timestamp validation)
4. Verify your account has the necessary permissions

**Solution Example**:
```python
# Correct authentication headers
import time
import hmac
import hashlib
import base64

def generate_auth_headers(access_key, secret_key, method, path):
    timestamp = str(int(time.time()))
    message = f"{method}\n{path}\n{timestamp}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    
    return {
        "X-Within-Access-Key": access_key,
        "X-Within-Timestamp": timestamp,
        "X-Within-Signature": base64.b64encode(signature).decode()
    }
```

### Rate Limiting

**Problem**: API requests are being rate limited (429 Too Many Requests).

**Troubleshooting Steps**:
1. Check the rate limits for the endpoints you're calling
2. Implement exponential backoff and retry logic
3. Consider batching requests when possible
4. Monitor rate limit headers in responses to adjust your request rate

**Solution Example**:
```python
def make_api_request_with_backoff(url, headers, max_retries=5, base_delay=1):
    retries = 0
    while retries <= max_retries:
        response = requests.get(url, headers=headers)
        
        # Check if rate limited
        if response.status_code == 429:
            # Get retry time from header or use exponential backoff
            retry_after = int(response.headers.get('Retry-After', 2 ** retries))
            print(f"Rate limited. Retrying after {retry_after} seconds")
            time.sleep(retry_after)
            retries += 1
            continue
            
        # Return successful response
        return response
    
    raise Exception(f"Failed after {max_retries} retries")
```

### Invalid Request Format

**Problem**: Requests fail with 400 Bad Request errors.

**Troubleshooting Steps**:
1. Check API documentation for correct request format
2. Validate JSON payload for syntax errors
3. Ensure all required parameters are included
4. Verify parameter data types match API expectations

**Solution Example**:
```python
# Validate request body before sending
import json
import jsonschema

# API request schema
ad_score_schema = {
    "type": "object",
    "required": ["ad_content", "platform"],
    "properties": {
        "ad_content": {
            "type": "string",
            "minLength": 1
        },
        "platform": {
            "type": "string",
            "enum": ["facebook", "google", "tiktok"]
        }
    }
}

# Validate request
request_data = {
    "ad_content": "Limited time offer!",
    "platform": "facebook"
}

try:
    jsonschema.validate(instance=request_data, schema=ad_score_schema)
    print("Request data is valid")
except jsonschema.exceptions.ValidationError as e:
    print(f"Invalid request data: {e}")
```

## Model Performance Issues

### Unexpected Prediction Results

**Problem**: Model predictions differ significantly from expectations.

**Troubleshooting Steps**:
1. Verify input data matches the expected format and ranges
2. Check for data preprocessing inconsistencies
3. Ensure you're using the correct model version
4. Compare with example predictions from documentation

**Solution Example**:
```python
# Check input data ranges and formats
def validate_model_input(input_data):
    issues = []
    
    # Check ad content length
    if len(input_data.get("ad_content", "")) < 10:
        issues.append("Ad content is too short for reliable prediction")
    
    # Check for valid platform
    valid_platforms = ["facebook", "google", "tiktok", "linkedin"]
    if input_data.get("platform") not in valid_platforms:
        issues.append(f"Invalid platform. Must be one of: {valid_platforms}")
    
    # Check numeric ranges if present
    if "historical_ctr" in input_data:
        if not (0 <= input_data["historical_ctr"] <= 1):
            issues.append("historical_ctr must be between 0 and 1")
    
    return issues
```

### Model Latency Issues

**Problem**: Model predictions are taking too long.

**Troubleshooting Steps**:
1. Check your network latency to the API
2. Verify the size of your request payloads
3. Consider batching multiple predictions
4. Check for any background processes competing for resources

**Solution Example**:
```python
# Measure prediction latency
import time

def measure_prediction_latency(client, ad_data, iterations=10):
    latencies = []
    
    for _ in range(iterations):
        start_time = time.time()
        _ = client.predict_ad_score(ad_data)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Min latency: {min_latency:.2f}ms")
    print(f"Max latency: {max_latency:.2f}ms")
    
    # Check if latency is above threshold
    if avg_latency > 500:  # 500ms threshold
        print("WARNING: Average latency is above recommended threshold")
        return False
    return True
```

### Feature Importance Discrepancies

**Problem**: Feature importance values don't match expectations.

**Troubleshooting Steps**:
1. Verify input feature distributions match training data
2. Check for feature preprocessing consistency
3. Understand that feature importance can vary between model versions
4. Consider that feature correlations in your data might differ from training data

**Solution Example**:
```python
# Analyze feature distributions to find issues
import numpy as np
import matplotlib.pyplot as plt

def analyze_feature_distribution(feature_values, feature_name, expected_range=None):
    """Analyze feature distribution for potential issues"""
    # Basic statistics
    mean_val = np.mean(feature_values)
    median_val = np.median(feature_values)
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    std_val = np.std(feature_values)
    
    print(f"Feature: {feature_name}")
    print(f"Mean: {mean_val:.4f}, Median: {median_val:.4f}")
    print(f"Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"Std Dev: {std_val:.4f}")
    
    # Check expected range if provided
    if expected_range:
        lower, upper = expected_range
        out_of_range = np.sum((feature_values < lower) | (feature_values > upper))
        percent_out = (out_of_range / len(feature_values)) * 100
        print(f"Values outside expected range [{lower}, {upper}]: {percent_out:.2f}%")
        if percent_out > 5:
            print("WARNING: Significant portion of values outside expected range")
    
    # Check for unusual distribution shapes
    skew = np.mean(((feature_values - mean_val) / std_val) ** 3)
    kurt = np.mean(((feature_values - mean_val) / std_val) ** 4) - 3
    print(f"Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")
    
    # Flag highly skewed or unusual distributions
    if abs(skew) > 2 or abs(kurt) > 7:
        print("WARNING: Distribution appears highly skewed or unusual")
```

### Model Drift

**Problem**: Model performance degrades over time.

**Troubleshooting Steps**:
1. Check if your data patterns have changed significantly
2. Monitor for feature distribution shifts
3. Consider retraining or updating your model version
4. Implement drift detection for early warnings

**Solution Example**:
```python
# Monitor for data drift
from scipy.stats import ks_2samp

def detect_feature_drift(reference_data, current_data, feature_name, threshold=0.05):
    """Detect drift in a feature using Kolmogorov-Smirnov test"""
    reference_values = reference_data[feature_name]
    current_values = current_data[feature_name]
    
    # Perform KS test
    statistic, p_value = ks_2samp(reference_values, current_values)
    
    drift_detected = p_value < threshold
    
    print(f"Feature: {feature_name}")
    print(f"KS statistic: {statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Drift detected: {drift_detected}")
    
    return {
        "feature": feature_name,
        "drift_detected": drift_detected,
        "statistic": statistic,
        "p_value": p_value
    }
```

## Data Integration Issues

### Data Import Failures

**Problem**: Data import jobs fail or produce errors.

**Troubleshooting Steps**:
1. Verify data file format matches expected schema
2. Check for encoding issues (use UTF-8 encoding)
3. Look for malformed records or invalid values
4. Ensure file permissions are set correctly

**Solution Example**:
```python
# Validate CSV data before import
import pandas as pd
import numpy as np

def validate_import_file(file_path, expected_columns):
    """Validate import file for common issues"""
    issues = []
    
    try:
        # Try to read the file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty file
        if len(df) == 0:
            issues.append("File contains no data rows")
        
        # Check for completely empty columns
        empty_columns = [col for col in df.columns if df[col].isna().all()]
        if empty_columns:
            issues.append(f"These columns contain only null values: {empty_columns}")
        
        # Check for invalid values in numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                issues.append(f"Column '{col}' contains missing values")
            
        return df, issues
        
    except Exception as e:
        return None, [f"Error reading file: {str(e)}"]

### Data Sync Inconsistencies

**Problem**: Data from advertising platforms is inconsistent or missing.

**Troubleshooting Steps**:
1. Verify API access to advertising platforms
2. Check for changes in platform API endpoints or data formats
3. Confirm correct date ranges are specified
4. Look for account permission issues

**Solution Example**:
```python
# Check data completeness across platforms
def check_data_completeness(data, date_range, platforms):
    """Check for data completeness across platforms and dates"""
    
    # Create a date range for expected data
    all_dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
    
    completeness_report = {}
    
    for platform in platforms:
        platform_data = data[data['platform'] == platform]
        
        # Get unique dates in the data
        existing_dates = pd.DatetimeIndex(platform_data['date'].unique())
        
        # Find missing dates
        missing_dates = all_dates.difference(existing_dates)
        
        # Calculate completeness percentage
        completeness_pct = 100 * (len(all_dates) - len(missing_dates)) / len(all_dates)
        
        completeness_report[platform] = {
            "completeness_pct": completeness_pct,
            "missing_dates": missing_dates.tolist(),
            "missing_count": len(missing_dates)
        }
        
        if len(missing_dates) > 0:
            print(f"WARNING: {platform} is missing data for {len(missing_dates)} dates")
            
    return completeness_report
```

### Inconsistent Metric Definitions

**Problem**: Metrics from different sources don't align.

**Troubleshooting Steps**:
1. Check for different metric definitions across platforms
2. Verify time zone settings for data collection
3. Look for attribution window differences
4. Ensure consistent currency and unit conversions

**Solution Example**:
```python
# Normalize metrics across platforms
def normalize_metrics(data, metric_mappings):
    """Normalize metrics from different platforms to consistent definitions"""
    normalized_data = data.copy()
    
    # Apply platform-specific transformations
    for platform in data['platform'].unique():
        platform_mask = data['platform'] == platform
        mappings = metric_mappings.get(platform, {})
        
        for target_metric, source_info in mappings.items():
            source_metric = source_info['source']
            transform_func = source_info.get('transform', lambda x: x)
            
            # Apply transformation
            if source_metric in data.columns:
                normalized_data.loc[platform_mask, target_metric] = \
                    transform_func(data.loc[platform_mask, source_metric])
    
    return normalized_data
```

### Historical Data Changes

**Problem**: Historical data values change unexpectedly.

**Troubleshooting Steps**:
1. Check for data restatements from advertising platforms
2. Verify attribution window settings
3. Look for delayed conversion tracking
4. Consider time zone differences in reporting

**Solution Example**:
```python
# Monitor for significant data changes
def detect_data_changes(previous_data, current_data, date_column, metric_columns, threshold=0.05):
    """Detect significant changes in previously loaded historical data"""
    # Merge datasets on common dimensions
    merged = previous_data.merge(
        current_data,
        on=['platform', 'campaign_id', date_column],
        suffixes=('_prev', '_curr')
    )
    
    changes_detected = False
    change_report = {}
    
    # Check each metric for significant changes
    for metric in metric_columns:
        prev_col = f"{metric}_prev"
        curr_col = f"{metric}_curr"
        
        # Calculate percent change
        merged['pct_change'] = (merged[curr_col] - merged[prev_col]) / merged[prev_col].replace(0, np.nan)
        
        # Find significant changes
        significant_changes = merged[abs(merged['pct_change']) > threshold]
        
        if len(significant_changes) > 0:
            changes_detected = True
            change_report[metric] = {
                "count": len(significant_changes),
                "mean_change": significant_changes['pct_change'].mean(),
                "max_change": significant_changes['pct_change'].max(),
                "example_changes": significant_changes.head(5)
            }
            
    return changes_detected, change_report
```

## Authentication Problems

### Account Suspension

**Problem**: Account has been suspended or deactivated.

**Troubleshooting Steps**:
1. Verify account status and permissions
2. Contact support for further assistance
3. Ensure all account activities are compliant with terms of service
4. Verify account access rights

**Solution Example**:
```python
# Check account status
def check_account_status(access_key):
    response = requests.get("https://api.within.co/api/v1/account", headers={"X-Within-Access-Key": access_key})
    if response.status_code == 403:
        print("Account suspended or deactivated")
    else:
        print("Account is active")
```

### Access Denial

**Problem**: Access to the system is denied.

**Troubleshooting Steps**:
1. Verify account status and permissions
2. Contact support for further assistance
3. Ensure all account activities are compliant with terms of service
4. Verify account access rights

**Solution Example**:
```python
# Check access denial
def check_access_denial(access_key):
    response = requests.get("https://api.within.co/api/v1/account", headers={"X-Within-Access-Key": access_key})
    if response.status_code == 403:
        print("Access denied")
    else:
        print("Access granted")
```

### API Key Issues

**Problem**: API keys don't work or return authentication errors.

**Troubleshooting Steps**:
1. Verify you're using the correct access key and secret key
2. Check if the API key has expired or been revoked
3. Ensure API keys have the necessary permissions
4. Verify the key is being transmitted correctly in request headers

**Solution Example**:
```python
# Test API key validity
def validate_api_key(access_key, secret_key):
    """Test if API key is valid by making a simple authenticated request"""
    
    # Generate auth headers
    auth_headers = generate_auth_headers(access_key, secret_key, "GET", "/api/v1/account")
    
    try:
        # Make request to account endpoint
        response = requests.get(
            "https://api.within.co/api/v1/account",
            headers=auth_headers
        )
        
        if response.status_code == 200:
            print("API key is valid")
            return True
        elif response.status_code == 401:
            print("API key is invalid or expired")
            return False
        else:
            print(f"Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error validating API key: {str(e)}")
        return False

### Session Expiration

**Problem**: User sessions expire unexpectedly.

**Troubleshooting Steps**:
1. Check session timeout settings
2. Implement token refresh logic
3. Verify correct handling of authentication cookies
4. Check for IP address changes during session

**Solution Example**:
```python
# Implement token refresh logic
def get_authenticated_client(access_key, secret_key, auto_refresh=True):
    """Get authenticated client with auto token refresh"""
    from within import Client
    
    client = Client(
        access_key=access_key,
        secret_key=secret_key
    )
    
    if auto_refresh:
        # Setup auto-refresh interceptor
        def token_refresh_interceptor(request):
            try:
                return request()
            except AuthenticationError as e:
                if "token expired" in str(e).lower():
                    print("Token expired, refreshing...")
                    client.refresh_token()
                    return request()
                else:
                    raise
        
        client.add_request_interceptor(token_refresh_interceptor)
    
    return client
```

### Permission Denied Errors

**Problem**: Access denied to certain resources or actions.

**Troubleshooting Steps**:
1. Verify user role and permissions
2. Check resource-specific access controls
3. Ensure the account has access to the requested feature
4. Look for organization or team-based restrictions

**Solution Example**:
```python
# Check permissions for specific resources
def check_permissions(client, resources):
    """Check if current credentials have access to specific resources"""
    
    permission_results = {}
    
    for resource in resources:
        try:
            # Attempt to access resource (read-only)
            result = client.check_access(resource)
            permission_results[resource] = {
                "has_access": True,
                "permissions": result.get("permissions", [])
            }
        except PermissionDeniedError:
            permission_results[resource] = {
                "has_access": False,
                "permissions": []
            }
        except Exception as e:
            permission_results[resource] = {
                "has_access": False,
                "error": str(e)
            }
    
    # Print permission summary
    for resource, result in permission_results.items():
        access_status = "✅ Access granted" if result["has_access"] else "❌ Access denied"
        print(f"{resource}: {access_status}")
        
        if result["has_access"] and "permissions" in result:
            print(f"  Permissions: {', '.join(result['permissions'])}")
    
    return permission_results
```

### Multi-Factor Authentication Issues

**Problem**: Problems with MFA verification.

**Troubleshooting Steps**:
1. Verify correct time sync on authentication devices
2. Check for expired backup codes
3. Ensure the correct authentication app is being used
4. Contact support for account recovery if needed

**Solution Example**:
```python
# Guide for MFA troubleshooting
def mfa_troubleshooting_guide():
    """Print troubleshooting steps for MFA issues"""
    
    print("MFA Troubleshooting Guide:")
    print("1. Verify your device's time is correct:")
    print("   - Enable automatic time synchronization")
    print("   - Time must be within 30 seconds of actual time")
    print()
    print("2. Try using backup codes:")
    print("   - Locate your backup codes provided during MFA setup")
    print("   - Enter a backup code instead of the MFA code")
    print()
    print("3. Common issues:")
    print("   - Using wrong authentication app")
    print("   - Reinstalling app without transferring accounts")
    print("   - Attempting to use expired codes")
    print()
    print("4. If still having issues:")
    print("   - Contact support at support@within.co")
    print("   - Include your username and when you last successfully logged in")
```

## Dashboard Problems

### Loading Issues

**Problem**: Dashboards are slow to load or fail to load.

**Troubleshooting Steps**:
1. Check your internet connection
2. Clear browser cache and cookies
3. Try a different browser
4. Reduce the selected date range
5. Disable browser extensions that might interfere

**Solution Example**:
```javascript
// Browser console diagnostics for dashboard loading
function diagnoseLoadingIssues() {
  console.log("Running dashboard loading diagnostics...");
  
  // Check resource loading times
  const resources = performance.getEntriesByType("resource");
  const slowResources = resources.filter(r => r.duration > 1000);
  
  console.log(`Total resources: ${resources.length}`);
  console.log(`Slow resources (>1s): ${slowResources.length}`);
  
  if (slowResources.length > 0) {
    console.log("Slow loading resources:");
    slowResources.forEach(r => {
      console.log(`${r.name}: ${r.duration.toFixed(0)}ms`);
    });
  }
  
  // Check for failed requests
  const failedResources = resources.filter(r => !r.responseEnd);
  if (failedResources.length > 0) {
    console.log("Failed resources:");
    failedResources.forEach(r => console.log(r.name));
  }
  
  // Memory usage
  if (performance.memory) {
    console.log(`Memory usage: ${(performance.memory.usedJSHeapSize / 1048576).toFixed(2)}MB / ${(performance.memory.jsHeapSizeLimit / 1048576).toFixed(2)}MB`);
  }
  
  // Navigation timing
  const timing = performance.timing;
  console.log(`Page load time: ${timing.loadEventEnd - timing.navigationStart}ms`);
  console.log(`DOM ready: ${timing.domComplete - timing.domLoading}ms`);
}
```

### Visualization Errors

**Problem**: Charts or visualizations display errors or no data.

**Troubleshooting Steps**:
1. Check if data exists for the selected filters and date range
2. Verify the data source connection is working
3. Try refreshing the dashboard data
4. Check for conflicts between applied filters

**Solution Example**:
```javascript
// Dashboard troubleshooting helper
function checkVisualizationData(widgetId) {
  const widget = document.getElementById(widgetId);
  if (!widget) {
    console.error(`Widget ${widgetId} not found`);
    return;
  }
  
  const dataState = widget._dataState || widget.getAttribute('data-state');
  console.log(`Widget ${widgetId} data state:`, dataState);
  
  // Check if widget has data
  const hasData = widget.querySelector('.no-data-message') === null;
  console.log(`Has data: ${hasData}`);
  
  // Check data points
  const dataPoints = widget.querySelectorAll('.data-point, .bar, .line-point');
  console.log(`Data points: ${dataPoints.length}`);
  
  // Check for error messages
  const errorElements = widget.querySelectorAll('.error-message, .warning-message');
  if (errorElements.length > 0) {
    console.log("Error messages found:");
    errorElements.forEach(el => console.log(el.textContent));
  }
  
  // Get applied filters
  const filters = window.dashboardState?.filters || {};
  console.log("Applied filters:", filters);
  
  // Check date range
  const dateRange = window.dashboardState?.dateRange || {};
  console.log("Date range:", dateRange);
}
```

### Filter Interactions

**Problem**: Dashboard filters produce unexpected results.

**Troubleshooting Steps**:
1. Clear all filters and apply them one at a time
2. Check for conflicting filter values
3. Verify date range selection
4. Check for case sensitivity in text filters

**Solution Example**:
```javascript
// Reset and diagnose dashboard filters
function resetAndDiagnoseFilters() {
  console.log("Current filters:", window.dashboardState?.filters);
  
  // Store current filters for reference
  const previousFilters = {...window.dashboardState?.filters};
  
  // Clear all filters (implementation depends on dashboard framework)
  if (typeof window.clearAllFilters === 'function') {
    window.clearAllFilters();
    console.log("All filters cleared");
  } else {
    console.log("Manual filter clearing required");
  }
  
  // Apply filters one by one to identify issues
  console.log("Try applying filters individually to identify problematic combinations");
  
  return {
    previousFilters,
    // Helper to apply a single filter
    applyFilter: function(name, value) {
      if (typeof window.applyFilter === 'function') {
        window.applyFilter(name, value);
        console.log(`Applied filter ${name}=${value}`);
      } else {
        console.log(`Manually apply filter ${name}=${value}`);
      }
    }
  };
}
```

### Export Problems

**Problem**: Dashboard exports fail or produce incorrect data.

**Troubleshooting Steps**:
1. Check the file format you're exporting to
2. Verify the size of the data being exported
3. Ensure you have permission to export data
4. Try exporting a smaller date range or fewer metrics

**Solution Example**:
```javascript
// Diagnose export issues
function checkExportCompatibility() {
  console.log("Checking export compatibility...");
  
  // Check data volume
  const dataPoints = document.querySelectorAll('.data-point, .bar, .line-point, td');
  console.log(`Approximate data points: ${dataPoints.length}`);
  
  if (dataPoints.length > 10000) {
    console.log("WARNING: Large data volume may cause export issues");
    console.log("Try reducing date range or filtering data");
  }
  
  // Check browser compatibility
  const isChrome = /Chrome/.test(navigator.userAgent);
  const isFirefox = /Firefox/.test(navigator.userAgent);
  const isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
  
  console.log("Browser compatibility:");
  console.log(`Chrome: ${isChrome ? "✅" : "❌"}`);
  console.log(`Firefox: ${isFirefox ? "✅" : "❌"}`);
  console.log(`Safari: ${isSafari ? "✅" : "❌"}`);
  
  if (isSafari) {
    console.log("Note: Some export formats have limited support in Safari");
  }
  
  // Check for ad blockers that might interfere with downloads
  const hasAdBlocker = document.getElementById('ad-block-detector') === null;
  if (hasAdBlocker) {
    console.log("WARNING: Ad blocker detected which may interfere with exports");
  }
}
```

## Performance Bottlenecks

### Slow API Response Times

**Problem**: API requests take too long to complete.

**Troubleshooting Steps**:
1. Check network latency between your application and the API
2. Monitor request payload sizes
3. Implement response caching for frequently accessed data
4. Consider batching multiple requests

**Solution Example**:
```python
# Implement response caching
import time
from functools import lru_cache
from datetime import datetime, timedelta

class CachedApiClient:
    """API client with caching for improved performance"""
    
    def __init__(self, base_client, cache_ttl=300):
        """
        Initialize cached client
        
        Args:
            base_client: The underlying API client
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.client = base_client
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
    
    def get_data(self, endpoint, params=None):
        """Get data with caching"""
        params = params or {}
        
        # Create cache key
        cache_key = f"{endpoint}:{str(sorted(params.items()))}"
        
        # Check cache
        if cache_key in self.cache:
            timestamp = self.cache_timestamps[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            
            if age < self.cache_ttl:
                print(f"Cache hit for {endpoint} (age: {age:.1f}s)")
                return self.cache[cache_key]
        
        # Cache miss or expired, make actual request
        print(f"Cache miss for {endpoint}, fetching fresh data")
        start_time = time.time()
        result = self.client.get_data(endpoint, params)
        end_time = time.time()
        
        # Update cache
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.now()
        
        print(f"Request completed in {(end_time - start_time) * 1000:.0f}ms")
        return result
    
    def clear_cache(self):
        """Clear the entire cache"""
        self.cache = {}
        self.cache_timestamps = {}
        print("Cache cleared")
```

### High Resource Utilization

**Problem**: Application consumes excessive CPU or memory.

**Troubleshooting Steps**:
1. Profile application to identify resource-intensive operations
2. Optimize data processing for large datasets
3. Implement pagination for large result sets
4. Consider more efficient data structures or algorithms

**Solution Example**:
```python
# Memory usage profiling
import os
import psutil
import tracemalloc
import gc

def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function"""
    
    # Force garbage collection
    gc.collect()
    
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Start memory tracking
    tracemalloc.start()
    start_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Get memory usage after execution
    end_mem = process.memory_info().rss / 1024 / 1024  # MB
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Print memory usage information
    print(f"Memory before: {start_mem:.2f} MB")
    print(f"Memory after: {end_mem:.2f} MB")
    print(f"Memory increase: {end_mem - start_mem:.2f} MB")
    print(f"Peak memory during execution: {peak / 1024 / 1024:.2f} MB")
    
    return result
```

### Database Query Performance

**Problem**: Database queries are slow or time out.

**Troubleshooting Steps**:
1. Check for missing indexes
2. Optimize JOIN operations
3. Limit result set size with pagination
4. Consider denormalization for common query patterns

**Solution Example**:
```python
# Query optimization helpers
from sqlalchemy import text
import time

def analyze_query_performance(engine, query, params=None):
    """Analyze query performance and provide optimization suggestions"""
    params = params or {}
    
    # Measure query time
    start_time = time.time()
    result = engine.execute(query, params)
    data = result.fetchall()
    end_time = time.time()
    
    # Get query plan (PostgreSQL)
    explain_query = f"EXPLAIN ANALYZE {query}"
    explain_result = engine.execute(text(explain_query), params)
    explain_output = [row[0] for row in explain_result]
    
    # Print performance statistics
    print(f"Query execution time: {(end_time - start_time) * 1000:.2f}ms")
    print(f"Rows returned: {len(data)}")
    
    # Look for optimization opportunities in the query plan
    issues = []
    
    # Check for sequential scans on large tables
    if any("Seq Scan" in line for line in explain_output):
        issues.append("Sequential scan detected - consider adding indexes")
    
    # Check for hash joins on large tables
    if any("Hash Join" in line for line in explain_output):
        issues.append("Hash join detected - verify join conditions are optimized")
    
    # Check for sorting operations
    if any("Sort" in line for line in explain_output):
        issues.append("Sorting operation detected - consider adding index for ORDER BY clause")
    
    # Print optimization suggestions
    if issues:
        print("\nOptimization suggestions:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo obvious optimization opportunities detected")
    
    # Print query plan for detailed analysis
    print("\nQuery Plan:")
    for line in explain_output:
        print(line)
    
    return data
```

### Concurrency and Throughput Limitations

**Problem**: System performance degrades under high concurrency.

**Troubleshooting Steps**:
1. Identify bottlenecks with load testing
2. Implement connection pooling
3. Consider horizontal scaling
4. Optimize resource-intensive operations

**Solution Example**:
```python
# Implement connection pooling
from concurrent.futures import ThreadPoolExecutor
import time
import random

def process_with_concurrency(items, process_func, max_workers=10, timeout=None):
    """Process items concurrently with connection pooling"""
    
    results = {}
    errors = {}
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_func, item): item_id 
            for item_id, item in items.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item, timeout=timeout):
            item_id = future_to_item[future]
            try:
                results[item_id] = future.result()
            except Exception as e:
                errors[item_id] = str(e)
    
    end_time = time.time()
    
    # Print performance statistics
    print(f"Processed {len(items)} items in {end_time - start_time:.2f} seconds")
    print(f"Success: {len(results)}, Errors: {len(errors)}")
    print(f"Average time per item: {(end_time - start_time) * 1000 / len(items):.2f}ms")
    
    return results, errors
```

## Common Error Codes

### API Error Codes

| Error Code | HTTP Status | Description | Troubleshooting |
|------------|-------------|-------------|----------------|
| `AUTHENTICATION_ERROR` | 401 | Authentication failure | Verify API keys, check signature generation, ensure system clock is synchronized |
| `PERMISSION_DENIED` | 403 | Insufficient permissions | Check user role and permissions, verify account access |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found | Verify resource ID, check if resource exists |
| `VALIDATION_ERROR` | 400 | Invalid request parameters | Check request format, ensure all required fields are provided |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Implement rate limiting, add backoff logic |
| `INTERNAL_ERROR` | 500 | Server-side error | Retry with exponential backoff, report to support if persistent |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable | Retry after delay, check system status |

### Model Error Codes

| Error Code | Description | Troubleshooting |
|------------|-------------|----------------|
| `MODEL_NOT_FOUND` | Model version not found | Verify model version, check for typos |
| `INVALID_INPUT` | Invalid model input | Check input format and data types |
| `PREDICTION_ERROR` | Error during prediction | Verify input data, check for edge cases |
| `FEATURE_MISSING` | Required feature missing | Ensure all required features are provided |
| `MODEL_OVERLOADED` | Model service overloaded | Retry with backoff, consider batching requests |
| `DRIFT_DETECTED` | Data drift detected | Review input data distribution, consider retraining |

### Data Error Codes

| Error Code | Description | Troubleshooting |
|------------|-------------|----------------|
| `DATA_FORMAT_ERROR` | Invalid data format | Check data schema, ensure correct format |
| `DATA_VALIDATION_ERROR` | Data validation failed | Verify data meets validation rules |
| `DATA_SOURCE_ERROR` | Error accessing data source | Check data source connection, verify credentials |
| `DATA_PROCESSING_ERROR` | Error during data processing | Check data pipeline logs, verify data integrity |
| `DATA_TOO_LARGE` | Data exceeds size limits | Reduce data size, batch processing |
| `DATA_TOO_OLD` | Data is outside valid time range | Verify date range parameters |

**Error Response Example**:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": [
      {
        "field": "ad_content",
        "issue": "required_field_missing"
      },
      {
        "field": "platform",
        "issue": "invalid_value",
        "allowed_values": ["facebook", "google", "tiktok", "linkedin"]
      }
    ]
  },
  "meta": {
    "request_id": "req_12345",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

**Error Handling Example**:
```python
def handle_api_error(response):
    """Handle API error responses with appropriate actions"""
    
    if response.status_code == 400:
        print("Invalid request:")
        error_data = response.json().get("error", {})
        
        # Print field-specific errors
        for detail in error_data.get("details", []):
            field = detail.get("field", "unknown")
            issue = detail.get("issue", "unknown")
            print(f"Field '{field}': {issue}")
            
            # Suggest fixes for common issues
            if issue == "required_field_missing":
                print(f"  Fix: Add the '{field}' field to your request")
            elif issue == "invalid_value":
                allowed = detail.get("allowed_values", [])
                print(f"  Fix: Use one of the allowed values: {allowed}")
        
        return "validation_error"
        
    elif response.status_code == 401:
        print("Authentication error. Check your API credentials.")
        return "auth_error"
        
    elif response.status_code == 403:
        print("Permission denied. Your account doesn't have access to this resource.")
        return "permission_error"
        
    elif response.status_code == 404:
        print("Resource not found. Check the identifier or path.")
        return "not_found"
        
    elif response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        print(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        return "rate_limited"
        
    elif response.status_code >= 500:
        print("Server error. Try again later or contact support.")
        return "server_error"
    
    else:
        print(f"Unexpected error: {response.status_code}")
        return "unknown_error"
```

## Logging and Debugging

### Enabling Debug Logging

Increase log verbosity to identify issues:

```python
# Configure detailed logging
import logging
import sys

def setup_debug_logging():
    """Set up detailed debug logging"""
    
    # Create logger
    logger = logging.getLogger('within')
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create file handler
    file_handler = logging.FileHandler('within_debug.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log system info
    logger.debug("Debug logging enabled")
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Platform: {sys.platform}")
    
    return logger

# Usage
logger = setup_debug_logging()
logger.debug("Starting API request")
```

### Request Tracing

Add request IDs for tracing issues through the system:

```python
# Implement request tracing
import uuid
import time
import contextlib

class RequestTracer:
    """Helper for tracing requests through the system"""
    
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.spans = {}
        self.current_span = None
    
    @contextlib.contextmanager
    def span(self, name):
        """Create a span within the trace"""
        span_id = str(uuid.uuid4())
        parent_id = self.current_span
        self.current_span = span_id
        
        start_time = time.time()
        
        print(f"[{self.trace_id}][{span_id}] Starting: {name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            self.spans[span_id] = {
                "name": name,
                "parent_id": parent_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration
            }
            
            print(f"[{self.trace_id}][{span_id}] Completed: {name} ({duration:.2f}ms)")
            self.current_span = parent_id
    
    def get_report(self):
        """Get a report of all spans in the trace"""
        return {
            "trace_id": self.trace_id,
            "spans": self.spans,
            "total_spans": len(self.spans)
        }
```

### Data Validation Debugging

Identify issues with input data:

```python
# Debug data validation issues
def debug_validation_failure(data, schema):
    """Debug why data fails validation against schema"""
    import jsonschema
    
    # Try to validate and catch the specific error
    try:
        jsonschema.validate(instance=data, schema=schema)
        print("Data is valid!")
        return True
    except jsonschema.exceptions.ValidationError as e:
        print("Validation error:")
        print(f"Failed at: {e.json_path}")
        print(f"Instance: {e.instance}")
        print(f"Error message: {e.message}")
        
        # Get the specific validation rule that failed
        if e.validator == 'required':
            missing_props = set(e.validator_value) - set(e.instance.keys())
            print(f"Missing required properties: {missing_props}")
        elif e.validator == 'type':
            print(f"Type error: Expected {e.validator_value}, got {type(e.instance).__name__}")
        elif e.validator == 'enum':
            print(f"Value not in allowed options: {e.validator_value}")
        
        return False
    except Exception as e:
        print(f"Other validation error: {str(e)}")
        return False
```

### Performance Analysis

Identify performance bottlenecks:

```python
# Function timing decorator
import time
import functools

def timeit(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000
        print(f"Function '{func.__name__}' took {execution_time:.2f}ms to execute")
        
        return result
    return wrapper

# Usage
@timeit
def predict_ad_score(client, ad_data):
    return client.predict_ad_score(ad_data)
```

## Support Resources

### Documentation Resources

Access comprehensive documentation:

- Technical Documentation: See `/docs` directory in your project
- SDK Documentation: See `Client` class docstrings or generated API docs
- Code Examples: See `/docs/examples` directory for working code examples
- Model Documentation: See `/docs/implementation/ml` for model details

### Getting Help

When you need additional support:

1. **Check Documentation First**
   - Review relevant documentation for your issue
   - Search for similar issues in troubleshooting documentation

2. **Check Log Files**
   - Application logs may contain valuable error information
   - System logs may reveal environment issues

3. **Contact Support**
   - Email: support@within.co
   - Include:
     - Detailed description of the issue
     - Steps to reproduce
     - Error messages or codes
     - Version information
     - Request IDs if available
     - Log excerpts

4. **Community Resources**
   - Visit the documentation resources for community guides
   - Check implementation examples in the codebase

### Providing Feedback

Help us improve:

- Email: feedback@within.co
- Include specific suggestions for improvement
- Report documentation errors or omissions 
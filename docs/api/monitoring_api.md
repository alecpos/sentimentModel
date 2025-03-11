# Monitoring API Reference

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This document provides detailed information about the WITHIN Monitoring API, which allows programmatic access to monitoring data, metrics, and alerting configurations for the WITHIN Ad Score & Account Health Predictor system.

## Overview

The Monitoring API enables:
- Retrieving model performance metrics
- Monitoring system health and resource utilization
- Accessing data quality metrics
- Managing alerts and notifications
- Viewing prediction logs and audit trails

## Authentication

All API requests must be authenticated using API keys. See the [Authentication Guide](/docs/api/authentication.md) for detailed information.

Include your API credentials in all requests:

```
X-Within-Access-Key: your_access_key
X-Within-Timestamp: current_unix_timestamp
X-Within-Signature: generated_signature
```

## Base URL

All API endpoints are relative to:

```
https://api.within.co/api/v1/monitoring
```

## API Endpoints

### Model Monitoring

#### Get Model Performance Metrics

```
GET /models/{model_name}
```

Retrieve performance metrics for a specific model.

**Path Parameters:**
- `model_name` (required): Name of the model (ad_score_predictor, account_health_predictor, ad_sentiment_analyzer)

**Query Parameters:**
- `start_date` (optional): Start date for metrics (YYYY-MM-DD)
- `end_date` (optional): End date for metrics (YYYY-MM-DD)
- `metrics` (optional): Comma-separated list of specific metrics to retrieve
- `granularity` (optional): Time granularity (hourly, daily, weekly, monthly)

**Example Request:**
```bash
curl -X GET "https://api.within.co/api/v1/monitoring/models/ad_score_predictor?start_date=2023-09-01&end_date=2023-09-30&metrics=accuracy,latency,drift&granularity=daily" \
  -H "X-Within-Access-Key: your_access_key" \
  -H "X-Within-Timestamp: 1696168496" \
  -H "X-Within-Signature: your_signature"
```

**Example Response:**
```json
{
  "data": {
    "model_name": "ad_score_predictor",
    "version": "2.1.0",
    "metrics": {
      "accuracy": {
        "current": 0.91,
        "trend": 0.02,
        "threshold": 0.85,
        "status": "healthy",
        "history": [
          {"date": "2023-09-01", "value": 0.89},
          {"date": "2023-09-02", "value": 0.90},
          {"date": "2023-09-03", "value": 0.91}
        ],
        "average": 0.90
      },
      "latency": {
        "current": 125,
        "trend": -5,
        "threshold": 300,
        "status": "healthy",
        "history": [
          {"date": "2023-09-01", "value": 130},
          {"date": "2023-09-02", "value": 128},
          {"date": "2023-09-03", "value": 125}
        ],
        "average": 127.67,
        "p95": 145,
        "p99": 180
      },
      "drift": {
        "detected": false,
        "score": 0.12,
        "threshold": 0.25,
        "features_with_drift": [],
        "history": [
          {"date": "2023-09-01", "value": 0.10},
          {"date": "2023-09-02", "value": 0.11},
          {"date": "2023-09-03", "value": 0.12}
        ]
      }
    }
  },
  "meta": {
    "request_id": "req_monitoring_123",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Get Model List

```
GET /models
```

Retrieve a list of all available models with their current status.

**Example Response:**
```json
{
  "data": {
    "models": [
      {
        "name": "ad_score_predictor",
        "version": "2.1.0",
        "status": "healthy",
        "last_updated": "2023-08-15T10:30:00Z",
        "prediction_count_24h": 12458,
        "accuracy": 0.91
      },
      {
        "name": "account_health_predictor",
        "version": "1.5.0",
        "status": "warning",
        "last_updated": "2023-07-22T14:45:00Z",
        "prediction_count_24h": 3621,
        "accuracy": 0.86
      },
      {
        "name": "ad_sentiment_analyzer",
        "version": "3.2.0",
        "status": "healthy",
        "last_updated": "2023-09-10T08:15:00Z",
        "prediction_count_24h": 8932,
        "accuracy": 0.89
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_124",
    "timestamp": "2023-10-01T12:35:56Z"
  }
}
```

#### Get Model Prediction Log

```
GET /models/{model_name}/predictions
```

Retrieve prediction logs for a specific model.

**Query Parameters:**
- `start_date` (optional): Start date for logs (YYYY-MM-DD)
- `end_date` (optional): End date for logs (YYYY-MM-DD)
- `limit` (optional): Maximum number of logs to return (default: 100)
- `offset` (optional): Pagination offset
- `status` (optional): Filter by prediction status (success, error, warning)

**Example Response:**
```json
{
  "data": {
    "predictions": [
      {
        "id": "pred_12345",
        "model": "ad_score_predictor",
        "version": "2.1.0",
        "timestamp": "2023-09-30T14:32:15Z",
        "input_hash": "a1b2c3d4e5f6",
        "result": {
          "score": 85.4,
          "confidence": 0.92
        },
        "latency_ms": 127,
        "status": "success",
        "client_id": "client_789"
      },
      {
        "id": "pred_12346",
        "model": "ad_score_predictor",
        "version": "2.1.0",
        "timestamp": "2023-09-30T14:33:22Z",
        "input_hash": "f6e5d4c3b2a1",
        "result": {
          "score": 72.1,
          "confidence": 0.85
        },
        "latency_ms": 135,
        "status": "success",
        "client_id": "client_456"
      }
    ],
    "total": 12458,
    "limit": 100,
    "offset": 0
  },
  "meta": {
    "request_id": "req_monitoring_125",
    "timestamp": "2023-10-01T12:36:56Z"
  }
}
```

#### Get Feature Drift Analysis

```
GET /models/{model_name}/drift
```

Analyze feature drift for a specific model.

**Query Parameters:**
- `start_date` (optional): Start date for analysis
- `end_date` (optional): End date for analysis
- `features` (optional): Comma-separated list of features to analyze

**Example Response:**
```json
{
  "data": {
    "model_name": "ad_score_predictor",
    "analysis_period": {
      "start": "2023-09-01",
      "end": "2023-09-30"
    },
    "reference_period": {
      "start": "2023-08-01",
      "end": "2023-08-31"
    },
    "overall_drift_score": 0.12,
    "drift_detected": false,
    "drift_threshold": 0.25,
    "feature_drift": [
      {
        "feature": "ad_content.length",
        "drift_score": 0.05,
        "statistical_test": "ks_test",
        "p_value": 0.78,
        "distribution_change": "minimal"
      },
      {
        "feature": "historical_ctr",
        "drift_score": 0.18,
        "statistical_test": "ks_test",
        "p_value": 0.23,
        "distribution_change": "moderate"
      },
      {
        "feature": "platform_facebook",
        "drift_score": 0.07,
        "statistical_test": "chi_squared",
        "p_value": 0.68,
        "distribution_change": "minimal"
      }
    ],
    "recommendations": [
      "Monitor 'historical_ctr' feature for continuing drift",
      "No immediate retraining needed based on current drift levels"
    ]
  },
  "meta": {
    "request_id": "req_monitoring_126",
    "timestamp": "2023-10-01T12:37:56Z"
  }
}
```

### System Monitoring

#### Get System Health

```
GET /system/health
```

Get overall system health metrics.

**Example Response:**
```json
{
  "data": {
    "status": "healthy",
    "components": {
      "api": {
        "status": "healthy",
        "latency_p95_ms": 125,
        "error_rate_1h": 0.002,
        "request_rate_1m": 42.5
      },
      "database": {
        "status": "healthy",
        "connection_pool_usage": 0.35,
        "query_latency_p95_ms": 28,
        "replication_lag_s": 0.8
      },
      "prediction_service": {
        "status": "healthy",
        "latency_p95_ms": 145,
        "error_rate_1h": 0.001,
        "prediction_rate_1m": 18.2
      },
      "data_pipeline": {
        "status": "warning",
        "last_successful_run": "2023-09-30T12:00:00Z",
        "error_rate_24h": 0.05,
        "processing_time_trend": "increasing"
      }
    },
    "resource_utilization": {
      "cpu": 0.42,
      "memory": 0.65,
      "disk": 0.58,
      "network_in_mbps": 12.5,
      "network_out_mbps": 8.7
    },
    "incidents": {
      "active": 1,
      "resolved_24h": 2
    }
  },
  "meta": {
    "request_id": "req_monitoring_127",
    "timestamp": "2023-10-01T12:38:56Z"
  }
}
```

#### Get Resource Metrics

```
GET /system/resources
```

Get detailed resource utilization metrics.

**Query Parameters:**
- `start_date` (optional): Start date for metrics
- `end_date` (optional): End date for metrics
- `granularity` (optional): Time granularity (minute, hour, day)
- `resources` (optional): Specific resources to query (cpu, memory, disk, network)

**Example Response:**
```json
{
  "data": {
    "period": {
      "start": "2023-09-30T00:00:00Z",
      "end": "2023-10-01T00:00:00Z",
      "granularity": "hour"
    },
    "metrics": {
      "cpu": {
        "average": 0.42,
        "peak": 0.76,
        "history": [
          {"timestamp": "2023-09-30T00:00:00Z", "value": 0.35},
          {"timestamp": "2023-09-30T01:00:00Z", "value": 0.38},
          // Additional data points...
          {"timestamp": "2023-09-30T23:00:00Z", "value": 0.45}
        ]
      },
      "memory": {
        "average": 0.65,
        "peak": 0.82,
        "history": [
          {"timestamp": "2023-09-30T00:00:00Z", "value": 0.60},
          {"timestamp": "2023-09-30T01:00:00Z", "value": 0.62},
          // Additional data points...
          {"timestamp": "2023-09-30T23:00:00Z", "value": 0.70}
        ]
      },
      "disk": {
        "average": 0.58,
        "peak": 0.58,
        "history": [
          {"timestamp": "2023-09-30T00:00:00Z", "value": 0.57},
          {"timestamp": "2023-09-30T01:00:00Z", "value": 0.57},
          // Additional data points...
          {"timestamp": "2023-09-30T23:00:00Z", "value": 0.58}
        ]
      },
      "network": {
        "in_average_mbps": 12.5,
        "in_peak_mbps": 45.2,
        "out_average_mbps": 8.7,
        "out_peak_mbps": 32.6,
        "history": [
          {
            "timestamp": "2023-09-30T00:00:00Z", 
            "in_mbps": 10.2,
            "out_mbps": 7.5
          },
          // Additional data points...
          {
            "timestamp": "2023-09-30T23:00:00Z",
            "in_mbps": 14.8,
            "out_mbps": 9.3
          }
        ]
      }
    }
  },
  "meta": {
    "request_id": "req_monitoring_128",
    "timestamp": "2023-10-01T12:39:56Z"
  }
}
```

#### Get API Metrics

```
GET /system/api
```

Get API performance metrics.

**Query Parameters:**
- `start_date` (optional): Start date for metrics
- `end_date` (optional): End date for metrics
- `granularity` (optional): Time granularity (minute, hour, day)
- `endpoints` (optional): Specific endpoints to query

**Example Response:**
```json
{
  "data": {
    "period": {
      "start": "2023-09-30T00:00:00Z",
      "end": "2023-10-01T00:00:00Z",
      "granularity": "hour"
    },
    "overall": {
      "request_count": 125632,
      "error_count": 237,
      "error_rate": 0.0019,
      "latency_avg_ms": 87,
      "latency_p50_ms": 65,
      "latency_p95_ms": 125,
      "latency_p99_ms": 210
    },
    "by_endpoint": [
      {
        "endpoint": "/api/v1/ad-score/predict",
        "request_count": 42561,
        "error_count": 85,
        "error_rate": 0.002,
        "latency_avg_ms": 95,
        "latency_p95_ms": 145
      },
      {
        "endpoint": "/api/v1/account-health/score",
        "request_count": 12875,
        "error_count": 26,
        "error_rate": 0.002,
        "latency_avg_ms": 142,
        "latency_p95_ms": 210
      },
      {
        "endpoint": "/api/v1/nlp/sentiment",
        "request_count": 28743,
        "error_count": 43,
        "error_rate": 0.0015,
        "latency_avg_ms": 75,
        "latency_p95_ms": 120
      }
    ],
    "by_status_code": {
      "200": 124632,
      "400": 752,
      "401": 12,
      "404": 5,
      "429": 45,
      "500": 186
    }
  },
  "meta": {
    "request_id": "req_monitoring_129",
    "timestamp": "2023-10-01T12:40:56Z"
  }
}
```

### Data Quality Monitoring

#### Get Data Quality Metrics

```
GET /data-quality
```

Get data quality metrics across the system.

**Query Parameters:**
- `start_date` (optional): Start date for metrics
- `end_date` (optional): End date for metrics
- `data_sources` (optional): Specific data sources to query

**Example Response:**
```json
{
  "data": {
    "period": {
      "start": "2023-09-01",
      "end": "2023-09-30"
    },
    "overall_quality_score": 0.94,
    "metrics": {
      "completeness": {
        "score": 0.98,
        "trend": 0.01,
        "by_source": {
          "facebook": 0.99,
          "google": 0.97,
          "tiktok": 0.95
        }
      },
      "timeliness": {
        "score": 0.96,
        "trend": 0.0,
        "by_source": {
          "facebook": 0.98,
          "google": 0.97,
          "tiktok": 0.92
        }
      },
      "consistency": {
        "score": 0.95,
        "trend": 0.02,
        "by_source": {
          "facebook": 0.94,
          "google": 0.97,
          "tiktok": 0.91
        }
      },
      "validity": {
        "score": 0.99,
        "trend": 0.0,
        "by_source": {
          "facebook": 0.99,
          "google": 0.99,
          "tiktok": 0.98
        }
      }
    },
    "issues": [
      {
        "data_source": "tiktok",
        "issue_type": "timeliness",
        "description": "Increased data refresh latency",
        "detected_at": "2023-09-25T14:32:15Z",
        "status": "monitoring",
        "impact": "low"
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_130",
    "timestamp": "2023-10-01T12:41:56Z"
  }
}
```

#### Get Schema Validation Results

```
GET /data-quality/schema-validation
```

Get schema validation results for data sources.

**Example Response:**
```json
{
  "data": {
    "period": {
      "start": "2023-09-01",
      "end": "2023-09-30"
    },
    "validation_runs": 30,
    "overall_pass_rate": 0.98,
    "failures_by_source": {
      "facebook": 0,
      "google": 1,
      "tiktok": 2
    },
    "recent_failures": [
      {
        "data_source": "tiktok",
        "timestamp": "2023-09-28T15:45:22Z",
        "schema_version": "1.2.0",
        "issues": [
          {
            "field": "creative.format",
            "expected": "string enum",
            "found": "unknown format type",
            "records_affected": 14
          }
        ],
        "resolution_status": "fixed"
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_131",
    "timestamp": "2023-10-01T12:42:56Z"
  }
}
```

### Alert Management

#### List Alerts

```
GET /alerts
```

List all configured alerts.

**Example Response:**
```json
{
  "data": {
    "alerts": [
      {
        "id": "alert_123",
        "name": "Ad Score Accuracy Drop",
        "description": "Alert when Ad Score Predictor accuracy drops below threshold",
        "metric": "models.ad_score_predictor.accuracy",
        "condition": "< 0.85",
        "window": "1h",
        "severity": "warning",
        "channels": ["slack-ml-team", "email-ml-engineers"],
        "created_at": "2023-05-15T10:30:45Z",
        "last_triggered": "2023-09-15T08:22:12Z",
        "status": "active"
      },
      {
        "id": "alert_456",
        "name": "API High Latency",
        "description": "Alert when API latency exceeds threshold",
        "metric": "api.latency.p95",
        "condition": "> 300",
        "window": "5m",
        "severity": "critical",
        "channels": ["slack-ops", "pagerduty-oncall"],
        "created_at": "2023-04-10T14:22:33Z",
        "last_triggered": null,
        "status": "active"
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_132",
    "timestamp": "2023-10-01T12:43:56Z"
  }
}
```

#### Create Alert

```
POST /alerts
```

Create a new alert.

**Request Body:**
```json
{
  "name": "Data Pipeline Failure",
  "description": "Alert when data pipeline fails",
  "metric": "data_pipeline.status",
  "condition": "== 'failed'",
  "window": "15m",
  "severity": "high",
  "channels": ["slack-data-team", "email-data-engineers"],
  "notification_template": "Data pipeline {pipeline_name} has failed: {error_message}"
}
```

**Example Response:**
```json
{
  "data": {
    "id": "alert_789",
    "name": "Data Pipeline Failure",
    "status": "active",
    "created_at": "2023-10-01T12:44:56Z"
  },
  "meta": {
    "request_id": "req_monitoring_133",
    "timestamp": "2023-10-01T12:44:56Z"
  }
}
```

#### Update Alert

```
PUT /alerts/{alert_id}
```

Update an existing alert.

**Request Body:**
```json
{
  "condition": "== 'failed' OR == 'timeout'",
  "severity": "critical",
  "channels": ["slack-data-team", "email-data-engineers", "pagerduty-data"]
}
```

**Example Response:**
```json
{
  "data": {
    "id": "alert_789",
    "updated_at": "2023-10-01T12:45:56Z"
  },
  "meta": {
    "request_id": "req_monitoring_134",
    "timestamp": "2023-10-01T12:45:56Z"
  }
}
```

#### Delete Alert

```
DELETE /alerts/{alert_id}
```

Delete an alert.

**Example Response:**
```json
{
  "data": {
    "success": true,
    "message": "Alert deleted successfully"
  },
  "meta": {
    "request_id": "req_monitoring_135",
    "timestamp": "2023-10-01T12:46:56Z"
  }
}
```

#### Get Alert History

```
GET /alerts/{alert_id}/history
```

Get alert trigger history.

**Example Response:**
```json
{
  "data": {
    "alert_id": "alert_123",
    "triggers": [
      {
        "id": "trigger_12345",
        "timestamp": "2023-09-15T08:22:12Z",
        "metric_value": 0.83,
        "threshold": 0.85,
        "condition_met": true,
        "notifications_sent": 2,
        "resolved_at": "2023-09-15T09:45:30Z",
        "resolution_time_minutes": 83
      },
      {
        "id": "trigger_12346",
        "timestamp": "2023-08-28T14:35:22Z",
        "metric_value": 0.84,
        "threshold": 0.85,
        "condition_met": true,
        "notifications_sent": 2,
        "resolved_at": "2023-08-28T15:20:45Z",
        "resolution_time_minutes": 45
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_136",
    "timestamp": "2023-10-01T12:47:56Z"
  }
}
```

### Notification Channels

#### List Notification Channels

```
GET /notification-channels
```

List available notification channels.

**Example Response:**
```json
{
  "data": {
    "channels": [
      {
        "id": "slack-ml-team",
        "type": "slack",
        "name": "ML Team Slack Channel",
        "created_at": "2023-04-10T14:22:33Z",
        "configuration": {
          "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
          "channel": "#ml-alerts"
        },
        "status": "active"
      },
      {
        "id": "email-ml-engineers",
        "type": "email",
        "name": "ML Engineers Email Group",
        "created_at": "2023-04-10T14:25:12Z",
        "configuration": {
          "recipients": ["ml-team@example.com"]
        },
        "status": "active"
      },
      {
        "id": "pagerduty-oncall",
        "type": "pagerduty",
        "name": "On-Call PagerDuty Integration",
        "created_at": "2023-04-12T09:15:45Z",
        "configuration": {
          "integration_key": "xxxxxxxxxxxx",
          "service_id": "PXXXXXX"
        },
        "status": "active"
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_137",
    "timestamp": "2023-10-01T12:48:56Z"
  }
}
```

#### Create Notification Channel

```
POST /notification-channels
```

Create a new notification channel.

**Request Body:**
```json
{
  "type": "slack",
  "name": "Data Team Slack Channel",
  "configuration": {
    "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
    "channel": "#data-alerts"
  }
}
```

**Example Response:**
```json
{
  "data": {
    "id": "slack-data-team",
    "type": "slack",
    "name": "Data Team Slack Channel",
    "created_at": "2023-10-01T12:49:56Z",
    "status": "active"
  },
  "meta": {
    "request_id": "req_monitoring_138",
    "timestamp": "2023-10-01T12:49:56Z"
  }
}
```

### Incident Management

#### List Incidents

```
GET /incidents
```

List all incidents.

**Query Parameters:**
- `status` (optional): Filter by status (active, resolved)
- `severity` (optional): Filter by severity (critical, high, medium, low)
- `start_date` (optional): Start date for incidents
- `end_date` (optional): End date for incidents

**Example Response:**
```json
{
  "data": {
    "incidents": [
      {
        "id": "inc_12345",
        "title": "Ad Score Predictor Accuracy Drop",
        "description": "Accuracy dropped below threshold due to data drift",
        "severity": "high",
        "status": "active",
        "detected_at": "2023-10-01T10:15:22Z",
        "source": "automated_alert",
        "source_id": "alert_123",
        "affected_components": ["ad_score_predictor"],
        "assignee": "ml-team",
        "updates": [
          {
            "timestamp": "2023-10-01T10:20:45Z",
            "message": "Investigation started",
            "author": "system"
          },
          {
            "timestamp": "2023-10-01T11:05:30Z",
            "message": "Identified data drift in ad_content feature",
            "author": "john.doe@example.com"
          }
        ]
      },
      {
        "id": "inc_12346",
        "title": "API Latency Spike",
        "description": "API latency increased significantly due to database slowdown",
        "severity": "critical",
        "status": "resolved",
        "detected_at": "2023-09-28T08:45:12Z",
        "resolved_at": "2023-09-28T10:30:45Z",
        "resolution_time_minutes": 105,
        "source": "automated_alert",
        "source_id": "alert_456",
        "affected_components": ["api", "database"],
        "assignee": "ops-team",
        "resolution": "Database indexes were rebuilt to improve query performance"
      }
    ]
  },
  "meta": {
    "request_id": "req_monitoring_139",
    "timestamp": "2023-10-01T12:50:56Z"
  }
}
```

#### Get Incident Details

```
GET /incidents/{incident_id}
```

Get detailed information about a specific incident.

**Example Response:**
```json
{
  "data": {
    "id": "inc_12345",
    "title": "Ad Score Predictor Accuracy Drop",
    "description": "Accuracy dropped below threshold due to data drift",
    "severity": "high",
    "status": "active",
    "detected_at": "2023-10-01T10:15:22Z",
    "source": "automated_alert",
    "source_id": "alert_123",
    "affected_components": ["ad_score_predictor"],
    "assignee": "ml-team",
    "metrics": {
      "accuracy": {
        "before": 0.91,
        "current": 0.83,
        "threshold": 0.85
      },
      "drift_score": {
        "before": 0.12,
        "current": 0.28,
        "threshold": 0.25
      }
    },
    "updates": [
      {
        "timestamp": "2023-10-01T10:20:45Z",
        "message": "Investigation started",
        "author": "system"
      },
      {
        "timestamp": "2023-10-01T11:05:30Z",
        "message": "Identified data drift in ad_content feature",
        "author": "john.doe@example.com"
      }
    ],
    "timeline": [
      {
        "timestamp": "2023-10-01T09:45:00Z",
        "event": "Drift detection job identified increased drift score",
        "details": "Drift score increased to 0.28 (threshold: 0.25)"
      },
      {
        "timestamp": "2023-10-01T10:15:22Z",
        "event": "Accuracy monitoring detected drop below threshold",
        "details": "Accuracy dropped to 0.83 (threshold: 0.85)"
      },
      {
        "timestamp": "2023-10-01T10:15:23Z",
        "event": "Alert triggered",
        "details": "Alert 'Ad Score Accuracy Drop' triggered"
      },
      {
        "timestamp": "2023-10-01T10:15:25Z",
        "event": "Incident created",
        "details": "System created incident based on alert"
      }
    ],
    "related_incidents": ["inc_11234", "inc_10987"],
    "recommended_actions": [
      "Investigate recent changes in ad content distribution",
      "Consider model retraining with latest data",
      "Check for any platform API changes in the last 48 hours"
    ]
  },
  "meta": {
    "request_id": "req_monitoring_140",
    "timestamp": "2023-10-01T12:51:56Z"
  }
}
```

#### Update Incident

```
PUT /incidents/{incident_id}
```

Update an incident.

**Request Body:**
```json
{
  "status": "investigating",
  "assignee": "jane.smith@example.com",
  "update": {
    "message": "Investigating correlation with recent Facebook API changes"
  }
}
```

**Example Response:**
```json
{
  "data": {
    "id": "inc_12345",
    "status": "investigating",
    "updated_at": "2023-10-01T12:52:56Z",
    "update_id": "update_567"
  },
  "meta": {
    "request_id": "req_monitoring_141",
    "timestamp": "2023-10-01T12:52:56Z"
  }
}
```

#### Resolve Incident

```
POST /incidents/{incident_id}/resolve
```

Resolve an incident.

**Request Body:**
```json
{
  "resolution": "Retrained model with latest data from Facebook",
  "root_cause": "Data drift due to Facebook ad format changes",
  "preventive_measures": [
    "Implement more frequent data drift detection",
    "Add automatic retraining trigger for significant drift"
  ]
}
```

**Example Response:**
```json
{
  "data": {
    "id": "inc_12345",
    "status": "resolved",
    "resolved_at": "2023-10-01T12:53:56Z",
    "resolution_time_minutes": 158
  },
  "meta": {
    "request_id": "req_monitoring_142",
    "timestamp": "2023-10-01T12:53:56Z"
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failure
- `403 Forbidden`: Permission denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

Error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid date range specified",
    "details": {
      "parameter": "start_date",
      "issue": "must be a valid date in YYYY-MM-DD format"
    }
  },
  "meta": {
    "request_id": "req_monitoring_error_123",
    "timestamp": "2023-10-01T12:54:56Z"
  }
}
```

## Rate Limits

The Monitoring API implements rate limiting to ensure system stability:

| Endpoint Category | Rate Limit |
|-------------------|------------|
| Read operations | 120 requests per minute |
| Write operations | 60 requests per minute |
| Alert operations | 30 requests per minute |

Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the rate limit resets

## Additional Resources

- [API Overview](/docs/api/overview.md)
- [Authentication Guide](/docs/api/authentication.md)
- [Monitoring Guide](/docs/maintenance/monitoring_guide.md)
- [Alerting Reference](/docs/maintenance/alerting_reference.md)
- [Error Codes Reference](/docs/api/error_codes.md) 
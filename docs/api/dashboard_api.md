# Dashboard API Documentation

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


This document provides comprehensive details on the WITHIN Dashboard API, allowing developers to programmatically interact with dashboards, access dashboard data, and create custom integrations.

## Overview

The Dashboard API enables:
- Retrieving dashboard data and visualizations
- Creating and updating dashboards programmatically
- Embedding dashboards in external applications
- Scheduling and managing dashboard exports
- Fetching dashboard metadata and settings

## Authentication

All Dashboard API requests require authentication using API keys.

### API Key Authentication

Include your API key in the request headers:

```
X-Within-Access-Key: your_access_key
X-Within-Timestamp: current_unix_timestamp
X-Within-Signature: generated_signature
```

The signature is generated using HMAC-SHA256:

```python
import time
import hmac
import hashlib
import base64

def generate_signature(secret_key, method, path, timestamp):
    message = f"{method}\n{path}\n{timestamp}"
    signature = hmac.new(
        secret_key.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

def generate_auth_headers(access_key, secret_key, method, path):
    timestamp = str(int(time.time()))
    signature = generate_signature(secret_key, method, path, timestamp)
    
    return {
        "X-Within-Access-Key": access_key,
        "X-Within-Timestamp": timestamp,
        "X-Within-Signature": signature
    }
```

## API Endpoints

### Dashboard Management

#### List Dashboards

```
GET /api/v1/dashboards
```

Returns a list of all accessible dashboards.

**Query Parameters:**
- `limit` (optional): Maximum number of results (default: 20)
- `offset` (optional): Pagination offset (default: 0)
- `type` (optional): Filter by dashboard type (system, custom, shared)

**Response:**
```json
{
  "data": {
    "dashboards": [
      {
        "id": "db_12345",
        "name": "Ad Performance Overview",
        "type": "system",
        "created_at": "2023-01-15T10:30:45Z",
        "updated_at": "2023-09-20T14:22:33Z",
        "owner": "user@example.com",
        "url": "https://app.within.co/dashboards/db_12345"
      },
      {
        "id": "db_67890",
        "name": "Custom Campaign Analysis",
        "type": "custom",
        "created_at": "2023-06-10T08:15:30Z",
        "updated_at": "2023-09-21T11:45:12Z",
        "owner": "user@example.com",
        "url": "https://app.within.co/dashboards/db_67890"
      }
    ],
    "total": 12,
    "limit": 20,
    "offset": 0
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Get Dashboard Details

```
GET /api/v1/dashboards/{dashboard_id}
```

Returns detailed information about a specific dashboard.

**Response:**
```json
{
  "data": {
    "id": "db_12345",
    "name": "Ad Performance Overview",
    "type": "system",
    "description": "Overview of advertising performance metrics across platforms",
    "created_at": "2023-01-15T10:30:45Z",
    "updated_at": "2023-09-20T14:22:33Z",
    "owner": "user@example.com",
    "widgets": [
      {
        "id": "widget_123",
        "type": "metric_card",
        "title": "Total Ad Spend",
        "position": {"x": 0, "y": 0, "width": 4, "height": 2},
        "data_source": "ad_performance",
        "settings": {
          "metric": "total_spend",
          "comparison_period": "previous_period"
        }
      },
      {
        "id": "widget_456",
        "type": "line_chart",
        "title": "Daily Spend Trend",
        "position": {"x": 4, "y": 0, "width": 8, "height": 4},
        "data_source": "ad_performance",
        "settings": {
          "metrics": ["daily_spend"],
          "dimensions": ["date"],
          "time_grain": "day"
        }
      }
    ],
    "layout": "grid",
    "settings": {
      "default_date_range": "last_30_days",
      "refresh_rate": 14400,
      "color_theme": "light"
    },
    "url": "https://app.within.co/dashboards/db_12345"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Create Dashboard

```
POST /api/v1/dashboards
```

Creates a new custom dashboard.

**Request Body:**
```json
{
  "name": "Campaign Performance Q4",
  "description": "Q4 2023 campaign performance tracking",
  "layout": "grid",
  "widgets": [
    {
      "type": "metric_card",
      "title": "Q4 Total Spend",
      "position": {"x": 0, "y": 0, "width": 4, "height": 2},
      "data_source": "ad_performance",
      "settings": {
        "metric": "total_spend",
        "filters": {
          "date": {"start": "2023-10-01", "end": "2023-12-31"}
        }
      }
    }
  ],
  "settings": {
    "default_date_range": "custom",
    "custom_date_range": {"start": "2023-10-01", "end": "2023-12-31"},
    "color_theme": "dark"
  }
}
```

**Response:**
```json
{
  "data": {
    "id": "db_98765",
    "name": "Campaign Performance Q4",
    "url": "https://app.within.co/dashboards/db_98765",
    "created_at": "2023-10-01T12:34:56Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Update Dashboard

```
PUT /api/v1/dashboards/{dashboard_id}
```

Updates an existing dashboard.

**Request Body:**
```json
{
  "name": "Campaign Performance Q4 2023",
  "description": "Updated description for Q4 campaigns",
  "widgets": [
    {
      "id": "widget_123",
      "title": "Q4 Total Spend Updated"
    }
  ],
  "settings": {
    "color_theme": "light"
  }
}
```

**Response:**
```json
{
  "data": {
    "id": "db_98765",
    "updated_at": "2023-10-01T13:45:12Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T13:45:12Z"
  }
}
```

#### Delete Dashboard

```
DELETE /api/v1/dashboards/{dashboard_id}
```

Deletes a dashboard.

**Response:**
```json
{
  "data": {
    "success": true,
    "message": "Dashboard deleted successfully"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T14:22:33Z"
  }
}
```

### Widget Management

#### List Dashboard Widgets

```
GET /api/v1/dashboards/{dashboard_id}/widgets
```

Returns a list of widgets in a dashboard.

**Response:**
```json
{
  "data": {
    "widgets": [
      {
        "id": "widget_123",
        "type": "metric_card",
        "title": "Total Ad Spend",
        "position": {"x": 0, "y": 0, "width": 4, "height": 2},
        "data_source": "ad_performance",
        "settings": {
          "metric": "total_spend",
          "comparison_period": "previous_period"
        }
      },
      {
        "id": "widget_456",
        "type": "line_chart",
        "title": "Daily Spend Trend",
        "position": {"x": 4, "y": 0, "width": 8, "height": 4},
        "data_source": "ad_performance",
        "settings": {
          "metrics": ["daily_spend"],
          "dimensions": ["date"],
          "time_grain": "day"
        }
      }
    ]
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Add Widget to Dashboard

```
POST /api/v1/dashboards/{dashboard_id}/widgets
```

Adds a new widget to a dashboard.

**Request Body:**
```json
{
  "type": "bar_chart",
  "title": "Platform Comparison",
  "position": {"x": 0, "y": 4, "width": 12, "height": 4},
  "data_source": "ad_performance",
  "settings": {
    "metrics": ["total_spend", "total_conversions"],
    "dimensions": ["platform"],
    "sort_by": "total_spend",
    "sort_order": "desc"
  }
}
```

**Response:**
```json
{
  "data": {
    "id": "widget_789",
    "type": "bar_chart",
    "title": "Platform Comparison"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Update Widget

```
PUT /api/v1/dashboards/{dashboard_id}/widgets/{widget_id}
```

Updates an existing widget.

**Request Body:**
```json
{
  "title": "Platform Spend Comparison",
  "position": {"x": 0, "y": 4, "width": 12, "height": 6},
  "settings": {
    "metrics": ["total_spend", "total_conversions", "roas"],
    "sort_by": "roas",
    "sort_order": "desc"
  }
}
```

**Response:**
```json
{
  "data": {
    "id": "widget_789",
    "updated_at": "2023-10-01T13:45:12Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T13:45:12Z"
  }
}
```

#### Delete Widget

```
DELETE /api/v1/dashboards/{dashboard_id}/widgets/{widget_id}
```

Removes a widget from a dashboard.

**Response:**
```json
{
  "data": {
    "success": true,
    "message": "Widget deleted successfully"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T14:22:33Z"
  }
}
```

### Dashboard Data

#### Get Widget Data

```
GET /api/v1/dashboards/{dashboard_id}/widgets/{widget_id}/data
```

Retrieves the data for a specific widget.

**Query Parameters:**
- `date_range` (optional): Predefined date range (last_7_days, last_30_days, etc.)
- `start_date` (optional): Custom start date (YYYY-MM-DD)
- `end_date` (optional): Custom end date (YYYY-MM-DD)
- `filters` (optional): JSON-encoded additional filters

**Response:**
```json
{
  "data": {
    "widget_id": "widget_789",
    "widget_type": "bar_chart",
    "results": [
      {
        "platform": "Facebook",
        "total_spend": 12500.45,
        "total_conversions": 325
      },
      {
        "platform": "Google",
        "total_spend": 9876.32,
        "total_conversions": 287
      },
      {
        "platform": "TikTok",
        "total_spend": 5432.10,
        "total_conversions": 143
      }
    ],
    "date_range": {
      "start": "2023-09-01",
      "end": "2023-09-30"
    }
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Export Dashboard Data

```
POST /api/v1/dashboards/{dashboard_id}/export
```

Exports dashboard data in specified format.

**Request Body:**
```json
{
  "format": "pdf",
  "paper_size": "letter",
  "orientation": "landscape",
  "include_filters": true,
  "date_range": "last_30_days"
}
```

**Response:**
```json
{
  "data": {
    "export_id": "export_12345",
    "status": "processing",
    "estimated_completion_time": "2023-10-01T12:39:56Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Get Export Status

```
GET /api/v1/exports/{export_id}
```

Checks the status of an export job.

**Response:**
```json
{
  "data": {
    "export_id": "export_12345",
    "status": "completed",
    "format": "pdf",
    "download_url": "https://storage.within.co/exports/dashboard_12345_20231001.pdf",
    "expiration_time": "2023-10-08T12:34:56Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:40:12Z"
  }
}
```

### Dashboard Sharing

#### Share Dashboard

```
POST /api/v1/dashboards/{dashboard_id}/share
```

Shares a dashboard with users.

**Request Body:**
```json
{
  "recipients": ["user@example.com", "anotheruser@example.com"],
  "permission": "view",
  "message": "Here's the Q4 performance dashboard we discussed.",
  "expires_at": "2023-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "data": {
    "share_id": "share_12345",
    "recipients": [
      {
        "email": "user@example.com",
        "status": "sent"
      },
      {
        "email": "anotheruser@example.com",
        "status": "sent"
      }
    ],
    "dashboard_id": "db_12345",
    "permission": "view",
    "expires_at": "2023-12-31T23:59:59Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

#### Get Share Settings

```
GET /api/v1/dashboards/{dashboard_id}/share
```

Returns sharing settings for a dashboard.

**Response:**
```json
{
  "data": {
    "shares": [
      {
        "share_id": "share_12345",
        "recipients": ["user@example.com", "anotheruser@example.com"],
        "permission": "view",
        "created_at": "2023-10-01T12:34:56Z",
        "expires_at": "2023-12-31T23:59:59Z",
        "created_by": "admin@company.com"
      }
    ],
    "public_link": null
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T13:45:12Z"
  }
}
```

#### Create Public Link

```
POST /api/v1/dashboards/{dashboard_id}/public-link
```

Creates a public link for dashboard access.

**Request Body:**
```json
{
  "enabled": true,
  "expires_at": "2023-12-31T23:59:59Z",
  "allow_filters": true,
  "password_protected": true,
  "password": "secure123"
}
```

**Response:**
```json
{
  "data": {
    "public_link_id": "pl_12345",
    "url": "https://app.within.co/public/dashboards/pl_12345",
    "enabled": true,
    "expires_at": "2023-12-31T23:59:59Z",
    "password_protected": true,
    "created_at": "2023-10-01T12:34:56Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

## Embedding Dashboards

### Generate Embed Token

```
POST /api/v1/dashboards/{dashboard_id}/embed-token
```

Generates a token for embedding a dashboard in external applications.

**Request Body:**
```json
{
  "allowed_domains": ["example.com", "app.example.com"],
  "expires_in": 86400,
  "filters": {
    "platform": "facebook",
    "campaign_id": "camp_12345"
  },
  "editable": false
}
```

**Response:**
```json
{
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "embed_url": "https://app.within.co/embed/dashboards/db_12345?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_at": "2023-10-02T12:34:56Z"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

### Embedding Code

Use the following HTML to embed a dashboard in your application:

```html
<iframe
  src="https://app.within.co/embed/dashboards/db_12345?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  width="100%"
  height="800"
  frameborder="0"
  allowfullscreen
></iframe>
```

## Scheduled Reports

### Create Schedule

```
POST /api/v1/dashboards/{dashboard_id}/schedules
```

Creates a scheduled report for a dashboard.

**Request Body:**
```json
{
  "name": "Weekly Performance Report",
  "frequency": "weekly",
  "day_of_week": 1,
  "time": "09:00:00",
  "timezone": "America/New_York",
  "format": "pdf",
  "recipients": ["team@example.com", "manager@example.com"],
  "subject": "Weekly Ad Performance Report",
  "message": "Here is the weekly performance report.",
  "filters": {
    "platform": "all"
  },
  "date_range": "last_7_days"
}
```

**Response:**
```json
{
  "data": {
    "schedule_id": "sched_12345",
    "name": "Weekly Performance Report",
    "next_execution": "2023-10-02T09:00:00-04:00",
    "dashboard_id": "db_12345"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

### List Schedules

```
GET /api/v1/dashboards/{dashboard_id}/schedules
```

Lists all scheduled reports for a dashboard.

**Response:**
```json
{
  "data": {
    "schedules": [
      {
        "schedule_id": "sched_12345",
        "name": "Weekly Performance Report",
        "frequency": "weekly",
        "day_of_week": 1,
        "time": "09:00:00",
        "timezone": "America/New_York",
        "format": "pdf",
        "recipients": ["team@example.com", "manager@example.com"],
        "created_at": "2023-10-01T12:34:56Z",
        "next_execution": "2023-10-02T09:00:00-04:00",
        "last_execution": null
      }
    ]
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T13:45:12Z"
  }
}
```

### Update Schedule

```
PUT /api/v1/dashboards/{dashboard_id}/schedules/{schedule_id}
```

Updates an existing schedule.

**Request Body:**
```json
{
  "name": "Weekly Performance Report - Updated",
  "recipients": ["team@example.com", "manager@example.com", "director@example.com"],
  "time": "10:00:00"
}
```

**Response:**
```json
{
  "data": {
    "schedule_id": "sched_12345",
    "updated_at": "2023-10-01T14:22:33Z",
    "next_execution": "2023-10-02T10:00:00-04:00"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T14:22:33Z"
  }
}
```

### Delete Schedule

```
DELETE /api/v1/dashboards/{dashboard_id}/schedules/{schedule_id}
```

Deletes a scheduled report.

**Response:**
```json
{
  "data": {
    "success": true,
    "message": "Schedule deleted successfully"
  },
  "meta": {
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T14:22:33Z"
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- `200 OK`: Request succeeded
- `201 Created`: Resource created successfully
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
    "request_id": "req_abcdef123456",
    "timestamp": "2023-10-01T12:34:56Z"
  }
}
```

## Rate Limits

The Dashboard API implements rate limiting to ensure system stability:

| Endpoint Category | Rate Limit |
|-------------------|------------|
| Read operations | 120 requests per minute |
| Write operations | 60 requests per minute |
| Export operations | 10 requests per minute |

Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Unix timestamp when the rate limit resets

## SDK Examples

### Python SDK

```python
from within import Client

# Initialize client
client = Client(
    access_key="your_access_key",
    secret_key="your_secret_key"
)

# List dashboards
dashboards = client.list_dashboards(limit=10)
for dashboard in dashboards:
    print(f"Dashboard: {dashboard['name']} (ID: {dashboard['id']})")

# Get data from a specific widget
widget_data = client.get_widget_data(
    dashboard_id="db_12345",
    widget_id="widget_789",
    date_range="last_30_days"
)

# Create a scheduled report
schedule = client.create_dashboard_schedule(
    dashboard_id="db_12345",
    name="Monthly Performance Report",
    frequency="monthly",
    day_of_month=1,
    time="08:00:00",
    timezone="UTC",
    format="pdf",
    recipients=["team@example.com"]
)
```

### JavaScript SDK

```javascript
const Within = require('within-js-client');

// Initialize client
const client = new Within.Client({
    accessKey: 'your_access_key',
    secretKey: 'your_secret_key'
});

// List dashboards
client.listDashboards({ limit: 10 })
    .then(dashboards => {
        dashboards.forEach(dashboard => {
            console.log(`Dashboard: ${dashboard.name} (ID: ${dashboard.id})`);
        });
    })
    .catch(error => console.error('Error:', error));

// Get data from a specific widget
client.getWidgetData({
    dashboardId: 'db_12345',
    widgetId: 'widget_789',
    dateRange: 'last_30_days'
})
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));

// Create a scheduled report
client.createDashboardSchedule({
    dashboardId: 'db_12345',
    name: 'Monthly Performance Report',
    frequency: 'monthly',
    dayOfMonth: 1,
    time: '08:00:00',
    timezone: 'UTC',
    format: 'pdf',
    recipients: ['team@example.com']
})
    .then(schedule => console.log(`Schedule created: ${schedule.scheduleId}`))
    .catch(error => console.error('Error:', error));
```

## Additional Resources

- [API Overview](/docs/api/overview.md)
- [Authentication Guide](/docs/api/authentication.md)
- [Dashboards User Guide](/docs/user_guides/dashboards.md)
- [Error Codes Reference](/docs/api/error_codes.md)
- [API Rate Limits](/docs/api/rate_limits.md) 
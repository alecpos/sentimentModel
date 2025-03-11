# Alerting Reference

**IMPLEMENTATION STATUS: IMPLEMENTED**


This document provides a comprehensive reference for the WITHIN alerting system, including alert types, configuration options, notification channels, and best practices for effective monitoring.

## Table of Contents

- [Alert Types](#alert-types)
- [Alert Configuration](#alert-configuration)
- [Notification Channels](#notification-channels)
- [Alert Conditions](#alert-conditions)
- [Alert Severity Levels](#alert-severity-levels)
- [Alert Templates](#alert-templates)
- [Best Practices](#best-practices)
- [Alert Examples](#alert-examples)
- [Troubleshooting](#troubleshooting)

## Alert Types

The WITHIN alerting system supports the following alert types:

### Model Performance Alerts

| Alert Type | Description | Default Threshold | Recommended Severity |
|------------|-------------|-------------------|----------------------|
| Accuracy Drop | Alerts when model accuracy drops below threshold | < 0.85 | Warning or High |
| Latency Spike | Alerts when model prediction latency exceeds threshold | > 300ms | Warning or High |
| Error Rate Increase | Alerts when model error rate exceeds threshold | > 1% | High |
| Drift Detected | Alerts when data drift exceeds threshold | > 0.25 | Warning |
| Feature Importance Shift | Alerts when feature importance shifts significantly | > 0.3 | Warning |
| Prediction Distribution Change | Alerts when prediction distribution changes significantly | > 0.2 | Warning |

### System Health Alerts

| Alert Type | Description | Default Threshold | Recommended Severity |
|------------|-------------|-------------------|----------------------|
| API Error Rate | Alerts when API error rate exceeds threshold | > 1% | High |
| API Latency | Alerts when API latency exceeds threshold | > 500ms | High |
| CPU Usage | Alerts when CPU usage exceeds threshold | > 85% | Warning or High |
| Memory Usage | Alerts when memory usage exceeds threshold | > 90% | Warning or High |
| Disk Usage | Alerts when disk usage exceeds threshold | > 85% | Warning or High |
| Queue Depth | Alerts when queue depth exceeds threshold | > 1000 | Warning |
| Service Down | Alerts when a critical service is down | Status != 'running' | Critical |

### Data Quality Alerts

| Alert Type | Description | Default Threshold | Recommended Severity |
|------------|-------------|-------------------|----------------------|
| Data Validation Failure | Alerts when data validation fails | Failure count > 0 | High |
| Data Freshness | Alerts when data is stale | Age > 1 hour | Warning or High |
| Schema Change | Alerts when data schema changes | Any change | Info or Warning |
| Missing Data | Alerts when data is missing for a period | Missing > 30 min | Warning or High |
| Data Pipeline Failure | Alerts when data pipeline job fails | Status = 'failed' | High |

### Business Metric Alerts

| Alert Type | Description | Default Threshold | Recommended Severity |
|------------|-------------|-------------------|----------------------|
| Account Health Score Drop | Alerts when account health score drops rapidly | Drop > 15 points | Warning or High |
| Campaign Performance Anomaly | Alerts on unusual campaign performance | Z-score > 3 | Warning |
| Budget Exhaustion Risk | Alerts when budget is at risk of early exhaustion | Risk > 80% | Warning |
| Conversion Rate Drop | Alerts when conversion rate drops significantly | Drop > 20% | Warning or High |

## Alert Configuration

Alerts can be configured via the UI or programmatically through the API. Each alert requires the following parameters:

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Descriptive name for the alert | "Ad Score Accuracy Drop" |
| `metric` | Metric to monitor | "models.ad_score_predictor.accuracy" |
| `condition` | Alert condition | "< 0.85" |
| `window` | Time window for evaluation | "1h" |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `description` | Detailed description of the alert | Empty | "Alert when ad score accuracy drops below threshold" |
| `severity` | Alert severity level | "warning" | "high" |
| `channels` | Notification channels | Default channel | ["slack-ml-team", "email-ml-engineers"] |
| `notification_template` | Custom notification template | Standard template | "Accuracy dropped to {{value}} (threshold: {{threshold}})" |
| `evaluation_frequency` | How often to evaluate the condition | "1m" | "5m" |
| `recovery_threshold` | Threshold for recovery state | Same as alert threshold | "0.87" |
| `recovery_window` | Time window for recovery evaluation | Same as alert window | "15m" |
| `annotations` | Additional metadata for the alert | {} | {"owner": "ml-team", "documentation": "link/to/docs"} |
| `silenced` | Whether the alert is silenced | false | true |
| `silenced_until` | Timestamp until which the alert is silenced | null | "2023-10-15T00:00:00Z" |
| `enable_auto_recovery` | Whether to auto-resolve alerts | true | false |

## Notification Channels

The system supports the following notification channels:

### Email

Send notifications to individual recipients or distribution lists.

**Configuration:**
```json
{
  "type": "email",
  "name": "ML Team Email",
  "configuration": {
    "recipients": ["ml-team@company.com", "data-science@company.com"],
    "cc": ["manager@company.com"],
    "include_details": true,
    "include_graphs": true
  }
}
```

### Slack

Send notifications to Slack channels or direct messages.

**Configuration:**
```json
{
  "type": "slack",
  "name": "ML Team Slack Channel",
  "configuration": {
    "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
    "channel": "#ml-alerts",
    "mention_users": ["@john", "@jane"],
    "mention_groups": ["@data-team"],
    "include_graphs": true
  }
}
```

### PagerDuty

Create incidents in PagerDuty for critical alerts.

**Configuration:**
```json
{
  "type": "pagerduty",
  "name": "On-Call Rotation",
  "configuration": {
    "integration_key": "abcdef123456",
    "service_id": "PXXXXXX",
    "severity_mapping": {
      "critical": "P1",
      "high": "P2",
      "warning": "P3",
      "info": "P4"
    }
  }
}
```

### Webhook

Send notifications to a custom webhook endpoint.

**Configuration:**
```json
{
  "type": "webhook",
  "name": "Custom System Integration",
  "configuration": {
    "url": "https://api.example.com/webhooks/alerts",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer token123",
      "Content-Type": "application/json"
    },
    "include_details": true
  }
}
```

### SMS

Send SMS notifications for critical alerts.

**Configuration:**
```json
{
  "type": "sms",
  "name": "Operations Team SMS",
  "configuration": {
    "phone_numbers": ["+1234567890", "+0987654321"],
    "max_length": 160,
    "critical_only": true
  }
}
```

## Alert Conditions

Alert conditions define when an alert should be triggered. The system supports various condition operators:

### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `>` | Greater than | "> 0.9" |
| `>=` | Greater than or equal to | ">= 0.9" |
| `<` | Less than | "< 0.85" |
| `<=` | Less than or equal to | "<= 0.85" |
| `==` | Equal to | "== 'failed'" |
| `!=` | Not equal to | "!= 'running'" |

### Compound Conditions

You can create compound conditions using logical operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `AND` | Logical AND | "> 0.9 AND < 0.95" |
| `OR` | Logical OR | "< 0.85 OR > 0.98" |
| `NOT` | Logical NOT | "NOT == 'running'" |

### Rate of Change Conditions

Rate of change conditions compare current values to previous values:

| Function | Description | Example |
|----------|-------------|---------|
| `change_percent()` | Percentage change | "change_percent(5m) < -10" |
| `change_absolute()` | Absolute change | "change_absolute(1h) < -0.05" |
| `rate_of_change()` | Rate of change per minute | "rate_of_change(30m) < -0.001" |

### Statistical Functions

Statistical functions for advanced conditions:

| Function | Description | Example |
|----------|-------------|---------|
| `avg()` | Average over window | "avg(15m) < 0.85" |
| `min()` | Minimum over window | "min(15m) < 0.80" |
| `max()` | Maximum over window | "max(15m) > 0.95" |
| `count()` | Count of data points | "count(30m) < 100" |
| `sum()` | Sum over window | "sum(1h) > 1000" |
| `stddev()` | Standard deviation | "stddev(1h) > 0.1" |
| `diff()` | Difference between current and past | "diff(24h) < -10" |
| `percentile()` | Nth percentile | "percentile(95, 30m) > 500" |

## Alert Severity Levels

The system uses four severity levels for alerts:

| Severity | Use Case | Response Time | Notification Channels |
|----------|----------|---------------|------------------------|
| **Critical** | Service outages, severe performance degradation, or data loss | < 30 minutes | Slack, PagerDuty, SMS, Email |
| **High** | Significant issues requiring prompt attention | < 4 hours | Slack, Email, SMS (optional) |
| **Warning** | Potential issues requiring investigation | < 24 hours | Slack, Email |
| **Info** | Informational events | No action required | Slack, Email (digest) |

## Alert Templates

Notification templates determine the content of alert notifications. Templates support variable substitution using the `{{variable}}` syntax.

### Available Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{alert_name}}` | Name of the alert | "Ad Score Accuracy Drop" |
| `{{alert_id}}` | Alert ID | "alert_12345" |
| `{{severity}}` | Alert severity | "high" |
| `{{metric}}` | Alert metric | "models.ad_score_predictor.accuracy" |
| `{{value}}` | Current metric value | "0.82" |
| `{{threshold}}` | Alert threshold | "0.85" |
| `{{condition}}` | Alert condition | "< 0.85" |
| `{{timestamp}}` | Alert trigger time | "2023-10-01T12:34:56Z" |
| `{{trigger_id}}` | Trigger event ID | "trigger_67890" |
| `{{dashboard_url}}` | URL to relevant dashboard | "https://app.within.co/dashboards/..." |
| `{{graph_url}}` | URL to metric graph | "https://app.within.co/graphs/..." |
| `{{window}}` | Alert evaluation window | "1h" |
| `{{previous_value}}` | Previous metric value | "0.90" |
| `{{change}}` | Change from previous value | "-0.08" |
| `{{change_percent}}` | Percentage change from previous value | "-8.9%" |

### Example Templates

#### Standard Template

```
[{{severity}}] Alert: {{alert_name}}

Metric: {{metric}}
Current Value: {{value}}
Threshold: {{threshold}}
Triggered at: {{timestamp}}

Dashboard: {{dashboard_url}}
```

#### Detailed Performance Template

```
[{{severity}}] {{alert_name}} 

The {{metric}} metric has dropped to {{value}}, which is below the threshold of {{threshold}}.

Change: {{change_percent}} from previous value of {{previous_value}}

This alert was triggered at {{timestamp}} based on data from the last {{window}}.

Please investigate this performance degradation immediately.

View Dashboard: {{dashboard_url}}
View Graph: {{graph_url}}
```

#### Recovery Template

```
[RESOLVED] {{alert_name}}

The {{metric}} metric has recovered to {{value}}, which is above the recovery threshold of {{threshold}}.

The alert was active for {{duration}} before recovery.

View Dashboard: {{dashboard_url}}
```

## Best Practices

Follow these best practices for effective alerting:

### Alert Design

1. **Alert on symptoms, not causes**: Alert on user-impacting issues (e.g., high latency) rather than causes (e.g., high CPU)
2. **Minimize noise**: Focus on actionable alerts that require human intervention
3. **Set meaningful thresholds**: Avoid false positives and alert fatigue
4. **Group related alerts**: Avoid alert storms by grouping related issues
5. **Include clear context**: Provide enough information for responders to understand the issue

### Alert Thresholds

1. **Calibrate thresholds carefully**: Use historical data to determine normal ranges
2. **Implement dynamic thresholds**: Consider using adaptive thresholds that adjust to patterns
3. **Add buffer zones**: Set warning thresholds before critical levels are reached
4. **Test thresholds**: Validate thresholds before enabling in production
5. **Review and adjust**: Regularly review and adjust thresholds based on performance

### Notification Strategy

1. **Match channels to severity**: Use immediate channels (PagerDuty, SMS) only for critical issues
2. **Define escalation paths**: Establish clear escalation procedures for unresolved alerts
3. **Implement alert grouping**: Consolidate related alerts to reduce notification volume
4. **Use maintenance windows**: Silence alerts during planned maintenance
5. **Rotate on-call responsibilities**: Avoid alert fatigue by rotating on-call duties

### Ongoing Maintenance

1. **Audit alerts regularly**: Review and prune unnecessary alerts
2. **Document alert responses**: Create runbooks for common alert scenarios
3. **Review alert effectiveness**: Track false positives and missed issues
4. **Update alert definitions**: Adjust as system behavior changes
5. **Train new team members**: Ensure all team members understand the alerting system

## Alert Examples

### Model Performance Alert

```json
{
  "name": "Ad Score Accuracy Drop",
  "description": "Alert when Ad Score Predictor accuracy drops below threshold",
  "metric": "models.ad_score_predictor.accuracy",
  "condition": "< 0.85",
  "window": "1h",
  "severity": "high",
  "channels": ["slack-ml-team", "email-ml-engineers"],
  "notification_template": "The Ad Score Predictor accuracy has dropped to {{value}}, which is below the threshold of {{threshold}}. Please investigate potential data drift or model degradation.",
  "annotations": {
    "owner": "ml-team",
    "documentation": "https://company-wiki.com/ml/ad-score-troubleshooting"
  }
}
```

### System Health Alert

```json
{
  "name": "API High Latency",
  "description": "Alert when API p95 latency exceeds threshold",
  "metric": "api.latency.p95",
  "condition": "> 300",
  "window": "5m",
  "severity": "critical",
  "channels": ["slack-ops", "pagerduty-oncall"],
  "notification_template": "API p95 latency has increased to {{value}}ms, exceeding the {{threshold}}ms threshold. This may impact user experience and prediction throughput.",
  "annotations": {
    "owner": "platform-team",
    "runbook": "https://company-wiki.com/ops/high-latency-runbook"
  }
}
```

### Data Quality Alert

```json
{
  "name": "Data Pipeline Failure",
  "description": "Alert when data pipeline job fails",
  "metric": "data_pipeline.status",
  "condition": "== 'failed'",
  "window": "15m",
  "severity": "high",
  "channels": ["slack-data-team", "email-data-engineers"],
  "notification_template": "The data pipeline {{value}} has failed. This may impact model predictions if not resolved quickly.",
  "annotations": {
    "owner": "data-team",
    "runbook": "https://company-wiki.com/data/pipeline-failure-recovery"
  }
}
```

### Composite Alert

```json
{
  "name": "Account Health Critical Issues",
  "description": "Alert when multiple account health issues are detected",
  "metric": "composite",
  "condition": "models.account_health_predictor.health_score < 50 AND models.account_health_predictor.anomalies > 3",
  "window": "1h",
  "severity": "high",
  "channels": ["slack-customer-success", "email-account-managers"],
  "notification_template": "Critical account health issues detected: Health score {{health_score}} with {{anomalies}} anomalies detected in the last hour.",
  "annotations": {
    "owner": "customer-success-team",
    "runbook": "https://company-wiki.com/cs/account-health-intervention"
  }
}
```

## Troubleshooting

### Missing Alerts

If alerts are not firing when expected:

1. Verify the alert is enabled and not silenced
2. Check that the metric exists and contains data
3. Validate the alert condition and threshold
4. Ensure the evaluation window is appropriate
5. Check notification channel configuration
6. Review alert history for any errors

### Alert Storms

If receiving too many similar alerts:

1. Implement alert grouping
2. Add dampening to reduce noise
3. Review and adjust thresholds
4. Add maintenance windows during known issues
5. Consider implementing alert dependencies

### Notification Failures

If alert notifications are not being received:

1. Check notification channel configuration
2. Verify recipient email addresses or Slack channels
3. Check for rate limiting or throttling
4. Ensure webhook endpoints are accessible
5. Check for authentication or authorization issues

### False Positives

If receiving alerts that are not indicative of real issues:

1. Adjust alert thresholds
2. Implement dampening to require sustained violations
3. Add more specific conditions
4. Consider using dynamic thresholds
5. Review and update alert definitions

## Additional Resources

- [Monitoring Guide](/docs/maintenance/monitoring_guide.md)
- [Monitoring API Reference](/docs/api/monitoring_api.md)
- [Incident Response Guide](/docs/maintenance/incident_response.md)
- [Runbook Creation Guide](/docs/maintenance/runbook_guide.md)
- [Dashboard Templates](/docs/resources/dashboard_templates) 
# Dashboard Templates

**IMPLEMENTATION STATUS: IMPLEMENTED**


This directory contains template configurations for common dashboard layouts in the WITHIN platform. These templates can be used as starting points for creating custom dashboards.

## Available Templates

- `ad_performance.json` - Template for monitoring ad performance across platforms
- `account_health.json` - Template for account health monitoring
- `campaign_analytics.json` - Template for campaign performance analytics
- `budget_tracking.json` - Template for budget utilization tracking
- `platform_comparison.json` - Template for comparing performance across platforms

## Usage

To use these templates:

1. Navigate to Dashboards > Custom Dashboards in the WITHIN platform
2. Click "Create New Dashboard"
3. Click "Import Template"
4. Select the desired template file

## Template Structure

Each template file is a JSON document with the following structure:

```json
{
  "name": "Template Name",
  "description": "Template description",
  "layout": "grid",
  "widgets": [
    {
      "type": "widget_type",
      "title": "Widget Title",
      "position": {"x": 0, "y": 0, "width": 4, "height": 2},
      "data_source": "data_source_name",
      "settings": {
        // Widget-specific settings
      }
    }
    // Additional widgets...
  ],
  "settings": {
    "default_date_range": "last_30_days",
    "refresh_rate": 14400,
    "color_theme": "light"
  }
}
```

## Customizing Templates

After importing a template, you can customize it by:

1. Adding, removing, or modifying widgets
2. Changing data sources
3. Adjusting layout and positioning
4. Updating dashboard settings

## Creating Your Own Templates

To create your own template:

1. Build a dashboard in the WITHIN platform
2. Click "Export Template" in the dashboard settings
3. Save the exported JSON file
4. Modify as needed for reuse

## Best Practices

For effective dashboard templates:

1. Focus on related metrics in a single dashboard
2. Use consistent visualization types for similar metrics
3. Organize widgets in a logical flow (most important at top)
4. Include descriptive titles and labels
5. Document any custom calculations or filters

For more guidance, see the [Dashboard Best Practices](/docs/user_guides/dashboard_best_practices.md) guide. 
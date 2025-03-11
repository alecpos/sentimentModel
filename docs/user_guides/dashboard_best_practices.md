# Dashboard Best Practices

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides recommendations and best practices for creating effective dashboards within the WITHIN platform. Following these guidelines will help you build informative, user-friendly dashboards that drive actionable insights.

## Dashboard Design Principles

### Focus on Key Metrics

- **Limit metrics per dashboard**: Include no more than 5-7 key metrics on a single dashboard
- **Prioritize actionable metrics**: Focus on metrics that drive decisions, not just interesting data
- **Use hierarchy**: Place the most important metrics at the top or in prominent positions
- **Include trend indicators**: Show whether metrics are improving or declining

### Visual Design

- **Maintain consistency**: Use consistent colors, fonts, and layouts across dashboards
- **Use appropriate visualizations**:
  - Bar charts for comparisons
  - Line charts for trends over time
  - Pie/donut charts for composition (limit to 5-7 segments)
  - Tables for detailed data
  - Heatmaps for showing patterns across two dimensions
- **Apply color purposefully**:
  - Use the WITHIN color palette for consistency
  - Reserve red/green for indicating negative/positive values
  - Ensure sufficient contrast for readability
  - Consider colorblind-friendly palettes

### Layout and Organization

- **Apply the F-pattern**: Place key information in the top and left areas
- **Group related metrics**: Keep related visualizations together
- **Use white space effectively**: Don't overcrowd dashboards
- **Consider mobile views**: Ensure dashboards work on smaller screens
- **Use appropriate sizing**: Size widgets based on their importance

## Performance Optimization

### Data Efficiency

- **Limit date ranges**: Default to showing the last 30 days of data instead of all history
- **Apply filters judiciously**: Too many filters can overcomplicate dashboards
- **Use aggregation**: Aggregate data at appropriate levels (daily, weekly, monthly)
- **Consider sampling**: For very large datasets, sample data for faster loading

### Query Optimization

- **Limit complex calculations**: Move complex calculations to data preparation when possible
- **Cache frequently used views**: Save optimized views of commonly accessed dashboards
- **Schedule refresh times**: Schedule data refreshes during off-peak hours

## Effective Dashboard Types

### Executive Dashboards

- **Purpose**: High-level overview for executives and stakeholders
- **Best Practices**:
  - Focus on KPIs and summary metrics
  - Include period-over-period comparisons
  - Provide clear context and benchmarks
  - Limit detail in favor of clear insights
  - Use annotations to highlight key events or changes

### Operational Dashboards

- **Purpose**: Day-to-day monitoring of business operations
- **Best Practices**:
  - Include real-time or near-real-time data
  - Focus on metrics requiring immediate action
  - Use thresholds and alerts for critical metrics
  - Include detailed breakdowns for troubleshooting
  - Design for frequent use and quick scanning

### Analytical Dashboards

- **Purpose**: Deep analysis of trends and patterns
- **Best Practices**:
  - Include more detailed visualizations
  - Provide interactive filtering capabilities
  - Include comparison features
  - Allow drilling down into detailed data
  - Support hypothesis testing

## Creating Custom Dashboards

### Planning

- **Define the purpose**: Clearly define what questions the dashboard should answer
- **Identify the audience**: Consider who will use the dashboard and their needs
- **Sketch the layout**: Plan the organization before building
- **Select appropriate metrics**: Choose metrics that serve the dashboard purpose
- **Define update frequency**: Determine how often the data should refresh

### Implementation

- **Start with a template**: Use existing templates as a starting point
- **Build iteratively**: Start with core metrics and add detail as needed
- **Test with users**: Get feedback from actual users
- **Document assumptions**: Note any data assumptions or limitations
- **Include context**: Add text explanations where needed

### Maintenance

- **Review regularly**: Evaluate dashboard effectiveness quarterly
- **Update as needed**: Remove unused metrics and add new relevant ones
- **Check performance**: Monitor dashboard loading times
- **Validate data**: Regularly verify data accuracy
- **Gather feedback**: Continuously collect user feedback

## Platform-Specific Dashboards

### Facebook Ads Dashboard

- **Key Metrics**: CTR, CPC, Frequency, Relevance Score, ROAS
- **Recommended Visuals**:
  - Campaign performance comparison (bar chart)
  - Daily/weekly trends (line chart)
  - Audience breakdown (pie chart)
  - Ad format performance (horizontal bar chart)

### Google Ads Dashboard

- **Key Metrics**: Quality Score, Impression Share, CTR, Conversion Rate, CPA
- **Recommended Visuals**:
  - Campaign performance trends (line chart)
  - Keyword performance (table)
  - Device breakdown (bar chart)
  - Geographic performance (map visualization)

### TikTok Ads Dashboard

- **Key Metrics**: Video View Rate, Engagement Rate, CPM, CTR, ROAS
- **Recommended Visuals**:
  - Creative performance comparison (bar chart)
  - Audience age/gender breakdown (horizontal bar chart)
  - Hourly engagement patterns (heatmap)
  - Content theme performance (bubble chart)

## Dashboard Sharing

### Scheduling Reports

- **Determine frequency**: Match report frequency to business cycles
- **Select appropriate format**: Choose PDF for presentation, Excel for analysis
- **Include context**: Add explanatory notes to scheduled reports
- **Target the right audience**: Only include relevant stakeholders

### Collaboration

- **Set clear permissions**: Define who can view vs. edit dashboards
- **Use commenting features**: Encourage discussion within the platform
- **Establish a review process**: Regularly review dashboard effectiveness
- **Document changes**: Keep a record of dashboard modifications

## Examples and Templates

### Example: Ad Performance Dashboard

```
[Top Row]
- Total Ad Spend KPI (with period comparison)
- Overall ROAS KPI (with period comparison)
- Total Conversions KPI (with period comparison)

[Middle Row]
- Ad Spend by Platform (bar chart)
- ROAS Trend by Week (line chart)
- Top Performing Campaigns (table)

[Bottom Row]
- Performance by Ad Format (horizontal bar chart)
- Geographic Performance (map)
- Device Breakdown (pie chart)
```

### Example: Account Health Dashboard

```
[Top Row]
- Account Health Score KPI (with trend indicator)
- Risk Level Indicator
- Top Recommendations

[Middle Row]
- Health Score Trend (line chart)
- Platform Comparison (radar chart)
- Key Metrics vs. Benchmarks (bullet charts)

[Bottom Row]
- Risk Factors Table
- Anomaly Detection Timeline
- Performance Forecast Chart
```

## Additional Resources

- [Dashboard Video Tutorials](https://learn.within.co/dashboards/tutorials)
- [WITHIN Design System Documentation](https://design.within.co)
- [Data Visualization Best Practices](https://within.co/blog/data-viz-best-practices)
- [Sample Dashboard Templates](https://community.within.co/dashboards/templates)
- [Dashboard API Documentation](/docs/api/dashboard_api.md)

For personalized dashboard guidance, contact your WITHIN account manager or [support@within.co](mailto:support@within.co). 
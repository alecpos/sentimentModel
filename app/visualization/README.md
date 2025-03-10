# Data Visualization

This directory contains the data visualization and reporting components for the WITHIN Ad Score & Account Health Predictor system. These components generate interactive visualizations, dashboards, and reports to help users understand predictions, model performance, and advertising insights.

## Directory Structure

```
visualization/
├── __init__.py                     # Visualization package initialization
├── charts/                         # Chart components
│   ├── __init__.py                 # Charts package initialization
│   ├── performance_charts.py       # Ad performance visualizations
│   ├── prediction_charts.py        # Prediction visualizations
│   ├── comparison_charts.py        # Comparison visualizations
│   └── explanatory_charts.py       # Model explanation visualizations
├── dashboards/                     # Dashboard components
│   ├── __init__.py                 # Dashboards package initialization
│   ├── ad_performance.py           # Ad performance dashboards
│   ├── account_health.py           # Account health dashboards
│   ├── model_monitoring.py         # Model monitoring dashboards
│   └── executive_summary.py        # Executive summary dashboards
├── reports/                        # Report templates
│   ├── __init__.py                 # Reports package initialization
│   ├── ad_analysis.py              # Ad analysis reports
│   ├── account_analysis.py         # Account analysis reports
│   ├── model_card.py               # Model card generator
│   └── prediction_report.py        # Prediction explanation reports
├── plots/                          # Plot utilities
│   ├── __init__.py                 # Plots package initialization
│   ├── colors.py                   # Color schemes
│   ├── themes.py                   # Plot themes
│   └── annotations.py              # Plot annotation utilities
└── utils/                          # Visualization utilities
    ├── __init__.py                 # Utils package initialization
    ├── data_preparation.py         # Data preparation for visualization
    ├── export.py                   # Export utilities (PDF, PNG, etc.)
    └── interactive.py              # Interactive visualization utilities
```

## Core Components

### Chart Components

The chart components create individual visualizations:

```python
from app.visualization.charts import PerformanceChart, ComparisonChart, PredictionChart

# Create a performance chart
performance_chart = PerformanceChart(
    title="Ad Performance Over Time",
    x_label="Date",
    y_label="Metrics",
    height=400,
    width=800
)

# Add data series
performance_chart.add_series(
    name="CTR",
    data=daily_ctr_data,
    color="#4285F4",
    y_axis="primary"
)
performance_chart.add_series(
    name="Conversions",
    data=daily_conversion_data,
    color="#34A853",
    y_axis="secondary"
)

# Generate chart
chart_html = performance_chart.render()
```

### Dashboard Components

The dashboard components combine multiple visualizations into cohesive dashboards:

```python
from app.visualization.dashboards import AdPerformanceDashboard

# Create a performance dashboard
dashboard = AdPerformanceDashboard(
    title="Campaign Performance Overview",
    time_period="Last 30 Days",
    campaign_id="camp_12345"
)

# Configure dashboard
dashboard.set_metrics([
    "impressions", "clicks", "conversions", "spend", "ctr", "cpa", "roas"
])
dashboard.add_comparison("previous_period")
dashboard.add_segment_breakdown(["platform", "ad_type", "audience"])

# Generate dashboard
dashboard_html = dashboard.render()
```

### Report Components

The report components generate comprehensive reports with visualizations and analysis:

```python
from app.visualization.reports import AdAnalysisReport

# Create an ad analysis report
report = AdAnalysisReport(
    title="Ad Performance Analysis",
    ad_ids=["ad_123", "ad_456", "ad_789"],
    time_period=("2023-01-01", "2023-03-31"),
    include_predictions=True,
    include_recommendations=True
)

# Generate report
report_html = report.generate()

# Export to PDF
report.export_pdf("ad_analysis_report.pdf")
```

## Visualization Types

### Performance Visualizations

Performance visualizations show advertising performance metrics:

1. **Time Series Charts**: Metrics over time
2. **Bar Charts**: Comparisons across segments
3. **Scatter Plots**: Relationship between metrics
4. **Heat Maps**: Performance by time or segment
5. **Funnel Charts**: Conversion funnels
6. **Gauge Charts**: Key performance indicators

### Prediction Visualizations

Prediction visualizations explain model predictions:

1. **Score Distributions**: Distribution of predicted scores
2. **Feature Importance**: SHAP or LIME visualizations
3. **Prediction Comparisons**: Actual vs. predicted
4. **Confidence Intervals**: Prediction uncertainty
5. **What-If Analysis**: Interactive prediction exploration

### Account Health Visualizations

Account health visualizations explain account status:

1. **Health Scorecards**: Overall account health scores
2. **Trend Indicators**: Performance trends over time
3. **Risk Factors**: Visualization of identified risks
4. **Opportunity Maps**: Areas for potential improvement
5. **Benchmark Comparisons**: Comparison to industry benchmarks

## Dashboard Types

### Ad Performance Dashboard

The Ad Performance Dashboard provides insights into ad effectiveness:

![Ad Performance Dashboard](https://placeholder-image-url.com/ad_performance_dashboard.png)

Components include:
- Performance summary cards
- Time series performance trends
- Performance by segment
- Top and bottom performing ads
- Performance distribution
- Platform comparison

### Account Health Dashboard

The Account Health Dashboard provides account-level insights:

![Account Health Dashboard](https://placeholder-image-url.com/account_health_dashboard.png)

Components include:
- Health score summary
- Historical health trends
- Risk factor visualization
- Opportunity prioritization
- Campaign performance overview
- Anomaly detection results

### Model Monitoring Dashboard

The Model Monitoring Dashboard shows model performance:

![Model Monitoring Dashboard](https://placeholder-image-url.com/model_monitoring_dashboard.png)

Components include:
- Performance metrics trends
- Prediction distribution
- Data drift visualization
- Feature importance stability
- Error analysis
- Fairness metrics

## Interactive Features

Dashboards include interactive features:

1. **Filtering**: Filter data by date range, platform, campaign, etc.
2. **Drill-down**: Click to explore detailed information
3. **Tooltips**: Hover for additional information
4. **What-if Analysis**: Adjust inputs to see impact on predictions
5. **Export**: Download visualizations or data
6. **Annotations**: Add notes to interesting findings

## Visualization Libraries

The visualization system leverages several libraries:

- **Plotly**: Interactive web-based visualizations
- **D3.js**: Custom, complex visualizations
- **Bokeh**: Server-backed interactive visualizations
- **Altair**: Declarative statistical visualizations
- **Matplotlib**: Static visualizations for reports
- **Seaborn**: Statistical visualizations
- **Dash**: Interactive dashboard applications

## Chart Design

All visualizations follow consistent design principles:

### Color Schemes

Standard color schemes ensure consistency:

```python
from app.visualization.plots import ColorSchemes

# Get standard colors
primary_colors = ColorSchemes.PRIMARY
brand_colors = ColorSchemes.BRAND
sequential_colors = ColorSchemes.SEQUENTIAL
diverging_colors = ColorSchemes.DIVERGING

# Apply colors to chart
chart.set_colors(primary_colors)
```

### Themes

Themes provide consistent styling:

```python
from app.visualization.plots import Themes

# Apply theme to visualization
chart.set_theme(Themes.LIGHT)
# or
chart.set_theme(Themes.DARK)
```

### Annotations

Annotations highlight important information:

```python
from app.visualization.plots import Annotations

# Add annotation to chart
Annotations.add_threshold(
    chart=performance_chart,
    value=0.05,
    label="Industry Benchmark CTR",
    line_style="dashed",
    color="#EA4335"
)
```

## Report Generation

The reporting system can generate various report types:

### Ad Analysis Report

The Ad Analysis Report provides detailed analysis of ad performance:

```python
from app.visualization.reports import AdAnalysisReport

# Generate comprehensive ad analysis
report = AdAnalysisReport(ad_id="ad_12345")
report.generate()
report.export_pdf("ad_analysis.pdf")
```

### Model Card

The Model Card provides documentation for ML models:

```python
from app.visualization.reports import ModelCardGenerator

# Generate model card
model_card = ModelCardGenerator(
    model_id="ad_score_model_v1",
    include_performance=True,
    include_fairness=True,
    include_limitations=True
)
model_card.generate()
model_card.export_html("model_card.html")
```

### Prediction Explanation

The Prediction Explanation report explains individual predictions:

```python
from app.visualization.reports import PredictionExplainer

# Generate explanation for a prediction
explainer = PredictionExplainer(prediction_id="pred_12345")
explanation = explainer.explain()
explanation.render()
```

## Export Options

Visualizations can be exported in various formats:

```python
from app.visualization.utils import export

# Export as PNG
export.to_png(chart, "chart.png", width=800, height=600)

# Export as SVG
export.to_svg(chart, "chart.svg")

# Export as PDF
export.to_pdf(dashboard, "dashboard.pdf")

# Export as interactive HTML
export.to_html(dashboard, "dashboard.html")

# Export underlying data
export.data_to_csv(chart, "chart_data.csv")
```

## Embedding in Applications

Visualizations can be embedded in applications:

```python
from app.visualization.utils import embed

# Get embeddable HTML
html_snippet = embed.get_embeddable_html(chart)

# Get iframe code
iframe_code = embed.get_iframe_code(dashboard_url, width=800, height=600)
```

## Development Guidelines

When enhancing or adding visualization components:

1. **Consistent Design**: Use the established color schemes and themes
2. **Responsive Design**: Ensure visualizations work on different devices
3. **Proper Labeling**: Include clear axis labels and titles
4. **Data Source Information**: Include data sources and timestamp information
5. **Hierarchical Dashboards**: Create overview-to-detail hierarchy
6. **Drill-down Capabilities**: Enable exploration of detailed information
7. **Appropriate Chart Types**: Use suitable chart types for different metrics
8. **Filtering and Comparison**: Include filtering and comparison features
9. **Feature Importance**: Visualize feature importance for model insights
10. **Performance Visualization**: Use standardized metrics across platforms
11. **Anomaly Highlighting**: Highlight outliers and significant changes
12. **Benchmark Comparisons**: Include relevant benchmarks when available 
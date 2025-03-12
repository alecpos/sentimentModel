"""
Analytics database models for the WITHIN ML Prediction System.

This module provides SQLAlchemy ORM models for analytics, reporting, and
visualization. These models handle persistence for report configurations,
saved queries, dashboards, and related analytics entities.

Key entities include:
- Report configurations and templates
- Saved queries and parameters
- Dashboard layouts and components
- Visualization settings
- Scheduled report definitions

DOCUMENTATION STATUS: COMPLETE
"""

__version__ = "0.1.0"  # Pre-release version

# Import base models
from app.core.db import BaseModel, FullModel, TimestampedModel

# Placeholder for future analytics model implementations
# from .report_model import Report, ReportTemplate
# from .dashboard_model import Dashboard, DashboardComponent
# from .query_model import SavedQuery, QueryParameter
# from .visualization_model import Visualization, VisualizationSetting
# from .schedule_model import ReportSchedule

# Export models
__all__ = [
    # List of model classes to export
    # "Report",
    # "ReportTemplate",
    # "Dashboard",
    # "DashboardComponent",
    # "SavedQuery", 
    # "QueryParameter",
    # "Visualization",
    # "VisualizationSetting",
    # "ReportSchedule"
] 
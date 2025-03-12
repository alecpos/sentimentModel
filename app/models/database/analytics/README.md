# Analytics Database Models

This directory contains SQLAlchemy database models for analytics and reporting in the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The analytics models provide database representations for:
- Configuring and persisting report definitions
- Managing dashboard layouts and components
- Storing and executing saved queries
- Configuring visualization settings
- Scheduling recurring reports

## Future Components

This directory is prepared for implementing the following database models:

### Report Models

Report configuration and templates:
- Report definitions and metadata
- Report parameters and filters
- Template definitions and layouts
- Report output formats
- Report versioning

### Dashboard Models

Dashboard configuration:
- Dashboard layouts and grids
- Dashboard components and widgets
- Permission settings
- Interactive element configuration
- Theme and styling settings

### Query Models

Saved query management:
- Query definitions and SQL/parameters
- Query metadata and descriptions
- Parameter definitions
- Query categorization
- Performance and usage metrics

### Visualization Models

Data visualization settings:
- Chart and graph configurations
- Visualization parameters
- Color schemes and styling
- Data transformation settings
- Interactive behavior configuration

### Schedule Models

Report scheduling:
- Scheduled report definitions
- Distribution lists and recipients
- Frequency and timing settings
- Delivery methods (email, API, etc.)
- Schedule status tracking

## Usage Example

Once implemented, analytics models would be used like this:

```python
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.database.analytics import (
    Report, 
    SavedQuery, 
    Dashboard, 
    ReportSchedule
)
from app.core.db import get_db

def generate_scheduled_reports():
    db = next(get_db())
    
    # Find reports scheduled for now
    now = datetime.utcnow()
    scheduled_reports = db.query(ReportSchedule)\
                         .filter(ReportSchedule.is_active == True)\
                         .filter(ReportSchedule.next_run_at <= now)\
                         .all()
    
    results = []
    for schedule in scheduled_reports:
        # Get the report definition
        report = db.query(Report).filter(Report.id == schedule.report_id).first()
        
        if not report:
            continue
            
        # Get the query for this report
        query = db.query(SavedQuery).filter(SavedQuery.id == report.query_id).first()
        
        if not query:
            continue
        
        # Execute the query with parameters
        # (simplified for example purposes)
        result = execute_query(query.query_text, report.parameters)
        
        # Update the schedule
        schedule.last_run_at = now
        schedule.next_run_at = calculate_next_run(schedule.frequency, now)
        db.add(schedule)
        
        # Create a result entry
        results.append({
            "report_id": report.id,
            "report_name": report.name,
            "executed_at": now,
            "result_count": len(result),
            "recipients": schedule.recipients
        })
    
    db.commit()
    return results
```

## Integration Points

- **Reporting System**: Uses these models for report generation
- **Admin Interface**: UI uses these models for report configuration
- **API Endpoints**: Report APIs use these models for data access
- **Export System**: Report export uses these configurations
- **Notification System**: Report scheduling triggers notifications

## Dependencies

- SQLAlchemy ORM for database operations
- Core DB components (FullModel with audit trails)
- JSON serialization for complex configuration
- Query execution engine
- Scheduling system for report execution 
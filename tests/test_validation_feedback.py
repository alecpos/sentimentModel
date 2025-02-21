import pytest
from datetime import datetime
from app.models.domain.data_catalog_model import DataCatalogModel
from app.core.feedback.feedback_handler import FeedbackProcessor

def test_data_quality_validation(db_session):
    """Test data quality validation functionality"""
    # Create test catalog entry
    entry = DataCatalogModel(
        name="Test Dataset",
        data_lake_id="dl_123",
        meta_info={
            "level": "campaign",
            "data_quality": {
                "completeness": 0.95,
                "accuracy": 0.98
            }
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    db_session.add(entry)
    db_session.commit()
    
    # Verify data quality metrics
    assert entry.meta_info["data_quality"]["completeness"] > 0.9
    assert entry.meta_info["data_quality"]["accuracy"] > 0.9

def test_feedback_processing(db_session):
    """Test feedback processing and resolution"""
    # Create test catalog entry
    entry = DataCatalogModel(
        name="Test Dataset",
        data_lake_id="dl_123",
        meta_info={
            "level": "campaign"
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    db_session.add(entry)
    db_session.commit()
    
    # Initialize feedback processor
    processor = FeedbackProcessor(db_session)
    
    # Add feedback
    result = processor.add_feedback(
        entry.id,
        "test_user",
        "Data quality needs improvement"
    )
    assert result["status"] == "success"
    assert result["feedback_count"] == 1
    
    # Get feedback stats
    stats = processor.get_feedback_stats(entry.id)
    assert stats["total_feedback"] == 1
    assert stats["pending_feedback"] == 1
    
    # Resolve feedback
    feedback_id = entry.feedback_log[0]["id"]
    resolution = processor.resolve_feedback(
        entry.id,
        feedback_id,
        "Improved data quality checks"
    )
    assert resolution["status"] == "success"
    
    # Verify resolution
    stats = processor.get_feedback_stats(entry.id)
    assert stats["resolved_feedback"] == 1
    assert stats["pending_feedback"] == 0
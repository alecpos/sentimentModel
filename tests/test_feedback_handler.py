import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from app.core.feedback.feedback_handler import FeedbackProcessor
from app.models.domain.data_catalog_model import DataCatalogModel
from app.models.domain.data_lake_model import DataLakeModel

@pytest.fixture
def feedback_processor(db_session):
    """Create feedback processor instance"""
    return FeedbackProcessor(db_session)

@pytest.fixture
def sample_catalog_entries(db_session):
    """Create sample catalog entries with feedback"""
    entries = []
    
    # Create data lake entry
    lake_entry = DataLakeModel(
        name="test_lake",
        data=b"test_data",
        meta_info={"level": "campaign", "data_layer": "raw"}
    )
    db_session.add(lake_entry)
    db_session.flush()

    # Create catalog entries with different feedback patterns
    for i in range(3):
        entry = DataCatalogModel(
            name=f"Test Dataset {i}",
            description=f"Test description {i}",
            data_lake_id=lake_entry.id,
            meta_info={"level": "campaign"},
            level_context={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        )
        entries.append(entry)
        db_session.add(entry)
    
    db_session.commit()
    return entries

class TestFeedbackProcessor:
    """Test suite for feedback processing functionality"""

    def test_add_feedback(self, feedback_processor, sample_catalog_entries):
        """Test adding feedback to catalog entry"""
        entry = sample_catalog_entries[0]
        
        result = feedback_processor.add_feedback(
            entry.id,
            "user123",
            "Need better documentation for field mappings"
        )
        
        assert result["status"] == "success"
        assert result["feedback_count"] == 1
        
        # Verify feedback structure
        feedback = entry.feedback_log[0]
        assert feedback["user_id"] == "user123"
        assert "timestamp" in feedback
        assert feedback["resolved"] is False

    def test_resolve_feedback(self, feedback_processor, sample_catalog_entries):
        """Test feedback resolution process"""
        entry = sample_catalog_entries[0]
        
        # Add feedback
        feedback_processor.add_feedback(
            entry.id,
            "user123",
            "Missing schema validation rules"
        )
        
        feedback_id = entry.feedback_log[0]["id"]
        original_version = entry.version
        
        # Resolve feedback
        result = feedback_processor.resolve_feedback(
            entry.id,
            feedback_id,
            "Added comprehensive schema validation"
        )
        
        assert result["status"] == "success"
        assert result["new_version"] > original_version
        
        # Verify resolution details
        feedback = entry.feedback_log[0]
        assert feedback["resolved"] is True
        assert "resolved_at" in feedback
        assert feedback["resolution_comment"] == "Added comprehensive schema validation"

    def test_get_pending_feedback(self, feedback_processor, sample_catalog_entries):
        """Test retrieval of pending feedback"""
        # Add mix of resolved and unresolved feedback
        entry = sample_catalog_entries[0]
        
        # Add two feedback items
        feedback_processor.add_feedback(entry.id, "user1", "Feedback 1")
        feedback_processor.add_feedback(entry.id, "user2", "Feedback 2")
        
        # Resolve one feedback item
        feedback_id = entry.feedback_log[0]["id"]
        feedback_processor.resolve_feedback(entry.id, feedback_id, "Resolution 1")
        
        # Get pending feedback
        pending = feedback_processor.get_pending_feedback()
        assert len(pending) == 1  # Only one unresolved feedback should remain
        
        # Verify pending feedback structure
        assert pending[0]["feedback"]["resolved"] is False
        assert pending[0]["feedback"]["comment"] == "Feedback 2"

    def test_feedback_stats(self, feedback_processor, sample_catalog_entries):
        """Test feedback statistics calculation"""
        entry = sample_catalog_entries[0]
        
        # Add multiple feedback items
        feedback_processor.add_feedback(entry.id, "user1", "Feedback 1")
        feedback_processor.add_feedback(entry.id, "user2", "Feedback 2")
        feedback_processor.add_feedback(entry.id, "user3", "Feedback 3")
        
        # Resolve some feedback
        feedback_id = entry.feedback_log[0]["id"]
        feedback_processor.resolve_feedback(entry.id, feedback_id, "Resolution 1")
        
        stats = feedback_processor.get_feedback_stats(entry.id)
        
        assert stats["total_feedback"] == 3
        assert stats["resolved_feedback"] == 1
        assert stats["pending_feedback"] == 2
        assert 0 < stats["resolution_rate"] < 1
        assert stats["last_feedback"] == entry.feedback_log[-1]

    def test_feedback_trends(self, feedback_processor, sample_catalog_entries):
        """Test feedback trend analysis"""
        entry = sample_catalog_entries[0]
        
        # Add three feedback items
        feedback_processor.add_feedback(entry.id, "user1", "Feedback 1")
        feedback_processor.add_feedback(entry.id, "user2", "Feedback 2")
        feedback_processor.add_feedback(entry.id, "user3", "Feedback 3")
        
        # Resolve one feedback
        feedback_id = entry.feedback_log[0]["id"]
        feedback_processor.resolve_feedback(entry.id, feedback_id, "Resolution 1")
        
        # Get trends
        trends = feedback_processor.analyze_feedback_trends()
        
        # Verify trend analysis
        assert trends["total_entries"] == len(sample_catalog_entries)
        assert trends["entries_with_feedback"] == 1
        assert trends["total_feedback"] == 3  # Total feedback items
        assert "avg_resolution_time" in trends

    def test_invalid_feedback_handling(self, feedback_processor):
        """Test handling of invalid feedback operations"""
        invalid_id = str(uuid4())
        
        # Test adding feedback to non-existent entry
        with pytest.raises(ValueError) as exc:
            feedback_processor.add_feedback(invalid_id, "user1", "test")
        assert "not found" in str(exc.value)
        
        # Test resolving feedback for non-existent entry
        with pytest.raises(ValueError) as exc:
            feedback_processor.resolve_feedback(invalid_id, "feedback_id", "resolution")
        assert "not found" in str(exc.value)

    def test_feedback_versioning(self, feedback_processor, sample_catalog_entries):
        """Test version management in feedback cycle"""
        entry = sample_catalog_entries[0]
        original_version = entry.version
        
        # Add and resolve multiple feedback items
        feedback_processor.add_feedback(entry.id, "user1", "Feedback 1")
        feedback_id1 = entry.feedback_log[0]["id"]
        feedback_processor.resolve_feedback(entry.id, feedback_id1, "Resolution 1")
        version1 = entry.version
        
        feedback_processor.add_feedback(entry.id, "user2", "Feedback 2")
        feedback_id2 = entry.feedback_log[1]["id"]
        feedback_processor.resolve_feedback(entry.id, feedback_id2, "Resolution 2")
        version2 = entry.version
        
        assert version1 > original_version
        assert version2 > version1

    def test_feedback_persistence(self, db_session):
        processor = FeedbackProcessor(db_session)
        entry = DataCatalogModel(
            name="Test Entry",  # Required
            data_lake_id=str(uuid4())  # Required
        )
        db_session.add(entry)
        db_session.commit()
        
        processor.add_feedback(entry.id, "user_123", "Dataset refresh frequency too low")
        assert len(entry.feedback_log) == 1
        assert entry.feedback_log[0]["comment"] == "Dataset refresh frequency too low"
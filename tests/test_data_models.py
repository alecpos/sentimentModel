# tests/test_data_models.py
import pytest
from datetime import datetime
from uuid import uuid4
from app.models.domain.data_catalog_model import DataCatalogModel
from app.models.domain.data_lake_model import DataLakeModel

@pytest.fixture
def sample_catalog_entry(db_session):
    """Create a sample catalog entry for testing"""
    lake_entry = DataLakeModel(
        name="test_lake_entry",
        data=b"test_data",
        meta_info={
            "level": "campaign",
            "data_layer": "raw"
        }
    )
    db_session.add(lake_entry)
    db_session.flush()

    entry = DataCatalogModel(
        name="Test Dataset",
        description="Test description",
        data_lake_id=lake_entry.id,
        schema_definition={
            "fields": [
                {"name": "id", "type": "string", "description": "Primary identifier"},
                {"name": "timestamp", "type": "datetime", "description": "Event timestamp"}
            ]
        },
        meta_info={
            "level": "campaign",
            "data_layer": "raw",
            "owner": "data_team",
            "sensitivity": "low"
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    return entry

class TestDataCatalogModel:
    """Test suite for DataCatalogModel functionality"""

    def test_basic_creation(self, db_session, sample_catalog_entry):
        """Test basic catalog entry creation with all required fields"""
        db_session.add(sample_catalog_entry)
        db_session.commit()

        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert retrieved.name == "Test Dataset"
        assert retrieved.schema_definition["fields"][0]["name"] == "id"
        assert retrieved.meta_info["level"] == "campaign"

    def test_multi_level_validation(self, db_session):
        """Test validation for different data levels"""
        test_cases = [
            {
                "level": "campaign",
                "context": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "target_demographic": "youth"
                },
                "should_pass": True
            },
            {
                "level": "user",
                "context": {
                    "user_segments": ["high_value", "active"],
                    "age_range": "18-25",
                    "geography": "US"
                },
                "should_pass": True
            },
            {
                "level": "campaign",
                "context": {
                    "start_date": "2024-01-01"  # Missing required fields
                },
                "should_pass": False
            }
        ]

        for case in test_cases:
            if case["should_pass"]:
                # Should create successfully
                entry = DataCatalogModel(
                    name="Test Entry",
                    data_lake_id=str(uuid4()),
                    meta_info={"level": case["level"]},
                    level_context=case["context"]
                )
                db_session.add(entry)
                db_session.commit()
                assert entry.id is not None
            else:
                # Should raise validation error
                with pytest.raises(ValueError) as exc_info:
                    entry = DataCatalogModel(
                        name="Test Entry",
                        data_lake_id=str(uuid4()),
                        meta_info={"level": case["level"]},
                        level_context=case["context"]
                    )
                assert "Missing required fields" in str(exc_info.value)

    def test_schema_documentation(self, db_session, sample_catalog_entry):
        """Test comprehensive schema documentation"""
        # Add detailed schema with documentation
        sample_catalog_entry.schema_definition = {
            "fields": [
                {
                    "name": "user_id",
                    "type": "string",
                    "description": "Unique user identifier",
                    "required": True,
                    "format": "UUID",
                    "examples": ["123e4567-e89b-12d3-a456-426614174000"]
                },
                {
                    "name": "event_timestamp",
                    "type": "datetime",
                    "description": "When the event occurred",
                    "required": True,
                    "format": "ISO-8601"
                }
            ],
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat(),
            "changelog": ["Initial schema definition"]
        }

        db_session.add(sample_catalog_entry)
        db_session.commit()

        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert len(retrieved.schema_definition["fields"]) == 2
        assert retrieved.schema_definition["fields"][0]["format"] == "UUID"
        assert "changelog" in retrieved.schema_definition

    def test_feedback_cycle(self, db_session, sample_catalog_entry):
        """Test feedback loop implementation"""
        db_session.add(sample_catalog_entry)
        db_session.commit()

        # Add feedback
        sample_catalog_entry.add_feedback(
            user_id="user123",
            comment="Need more detailed timestamp format documentation"
        )
        assert len(sample_catalog_entry.feedback_log) == 1
        
        # Resolve feedback with improvements
        feedback_id = sample_catalog_entry.feedback_log[0]["id"]
        original_version = sample_catalog_entry.version
        
        sample_catalog_entry.resolve_feedback(
            feedback_id,
            "Added timestamp format documentation and examples"
        )
        
        assert sample_catalog_entry.feedback_log[0]["resolved"] is True
        assert sample_catalog_entry.version > original_version

    def test_lineage_tracking(self, db_session, sample_catalog_entry):
        """Test data lineage documentation"""
        sample_catalog_entry.lineage = {
            "source": {
                "id": "camp_123",
                "name": "Campaign Data",
                "type": "raw_data",
                "system": "campaign_manager"
            },
            "transformations": [
                {
                    "type": "cleaning",
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Removed duplicate entries"
                },
                {
                    "type": "enrichment",
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Added geographic information"
                }
            ],
            "downstream_dependencies": [
                "analytics_dashboard",
                "monthly_reports"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }

        db_session.add(sample_catalog_entry)
        db_session.commit()

        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert len(retrieved.lineage["transformations"]) == 2
        assert "downstream_dependencies" in retrieved.lineage

    def test_usage_tracking(self, db_session, sample_catalog_entry):
        """Test usage statistics and documentation"""
        sample_catalog_entry.meta_info.update({
            "usage_stats": {
                "last_accessed": datetime.utcnow().isoformat(),
                "access_count": 42,
                "popular_queries": ["user_behavior", "conversion_rates"],
                "common_joins": ["user_profile", "campaign_metrics"]
            },
            "access_patterns": {
                "peak_hours": ["9-10", "14-15"],
                "typical_batch_size": "1000 records",
                "common_filters": ["date_range", "user_segment"]
            }
        })

        db_session.add(sample_catalog_entry)
        db_session.commit()

        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert "usage_stats" in retrieved.meta_info
        assert "access_patterns" in retrieved.meta_info

    def test_version_management(self, db_session, sample_catalog_entry):
        """Test version management and change tracking"""
        db_session.add(sample_catalog_entry)
        db_session.commit()

        # Simulate schema evolution
        original_version = sample_catalog_entry.version
        sample_catalog_entry.schema_definition["fields"].append({
            "name": "new_metric",
            "type": "float",
            "description": "New performance metric"
        })
        sample_catalog_entry._increment_version()

        assert sample_catalog_entry.version > original_version
        assert len(sample_catalog_entry.schema_definition["fields"]) == 3

class TestDataModels:
    """Test suite for validating core data model functionality"""
    
    def test_data_catalog_model_creation(self, db_session):
        """Verify DataCatalogModel can be properly created and persisted"""
        test_data = {
            "name": "Test Dataset",
            "description": "Test description",
            "data_lake_id": str(uuid4()),
            "meta_info": {
                "level": "campaign",
                "data_layer": "raw"
            },
            "level_context": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            },
            "schema_definition": {
                "fields": [
                    {"name": "id", "type": "uuid"},
                    {"name": "treatment", "type": "string"}
                ]
            }
        }
        
        # Create and persist entry
        entry = DataCatalogModel(**test_data)
        db_session.add(entry)
        db_session.commit()
        
        # Verify retrieval
        retrieved = db_session.get(DataCatalogModel, entry.id)
        assert retrieved.name == test_data["name"]
        assert len(retrieved.schema_definition["fields"]) == 2
        assert retrieved.schema_definition["fields"][0]["name"] == "id"
        assert retrieved.meta_info["level"] == "campaign"
        assert retrieved.level_context["target_demographic"] == "all"

    def test_data_lake_model_validation(self, db_session):
        """Test validation constraints for DataLakeModel"""
        # Test valid model creation
        valid_entry = DataLakeModel(
            name="User Activity Logs",
            data=b"user_activity_data",
            meta_info={
                "source": "mobile_app",
                "level": "user",
                "data_layer": "raw"
            }
        )
        db_session.add(valid_entry)
        db_session.commit()
        
        # Test name constraint validation
        with pytest.raises(AssertionError) as exc_info:
            invalid_entry = DataLakeModel(name="")  # Empty name
            db_session.add(invalid_entry)
            db_session.commit()
            
        assert "Name must be non-empty" in str(exc_info.value)
        
    def test_level_specific_metadata(self, db_session):
        """Verify level-specific context validation"""
        with pytest.raises(ValueError) as exc:
            invalid_entry = DataCatalogModel(
                meta_info={"level": "campaign"},
                level_context={"start_date": "2024-01-01"}  # Missing end_date
            )
            db_session.add(invalid_entry)
            db_session.commit()
        
        assert "Missing required fields" in str(exc.value)
# tests/test_data_catalog.py
from datetime import datetime
import pytest
from uuid import uuid4
from app.models.domain.data_catalog_model import DataCatalogModel
class TestDataCatalog:
    """Comprehensive test suite for Data Catalog functionality"""

    # Core Data Model Tests
    def test_basic_creation(self, db_session, sample_catalog_entry):
        """Test basic catalog entry creation"""
        db_session.add(sample_catalog_entry)
        db_session.commit()
        
        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert retrieved.name == "Test Dataset"
        assert retrieved.schema_definition["fields"][0]["name"] == "id"
        assert retrieved.meta_info["data_quality"]["completeness"] > 0.9

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
            }
        ]
        
        for case in test_cases:
            if case["should_pass"]:
                entry = DataCatalogModel(
                    name="Test Entry",
                    data_lake_id=str(uuid4()),
                    meta_info={"level": case["level"]},
                    level_context=case["context"]
                )
                db_session.add(entry)
                db_session.commit()
                assert entry.id is not None

    # Search Tests
    def test_search_functionality(self, search_service, mock_es):
        """Test basic and filtered search"""
        # Basic search
        results = search_service.search("test")
        assert len(results) == 1
        assert results[0]["name"] == "Test Dataset"

        # Filtered search
        results = search_service.search("campaign", filters={"level": "campaign"})
        assert len(results) == 1
        assert results[0]["meta_info"]["level"] == "campaign"

    def test_advanced_search(self, search_service, mock_es):
        """Test advanced search features"""
        # Faceted search
        results = search_service.search_with_facets("test", facets=["level", "data_layer"])
        assert "facets" in results
        
        # Compound queries
        results = search_service.search(
            "test",
            filters={
                "level": "campaign",
                "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
            }
        )
        assert len(results) > 0

    # Metadata Tests
    def test_metadata_documentation(self, db_session, sample_catalog_entry):
        """Test comprehensive metadata handling"""
        sample_catalog_entry.meta_info.update({
            "business_context": {
                "owner_team": "Data Science",
                "stakeholders": ["Marketing", "Analytics"],
                "use_cases": ["Campaign Optimization"],
                "sla_requirements": "99.9%"
            },
            "technical_context": {
                "source_systems": ["Campaign Manager"],
                "schema_version": "2.1",
                "data_retention": "12 months"
            }
        })
        
        db_session.add(sample_catalog_entry)
        db_session.commit()
        
        retrieved = db_session.get(DataCatalogModel, sample_catalog_entry.id)
        assert all(key in retrieved.meta_info for key in ["business_context", "technical_context"])

    # Lineage Tests
    def test_lineage_tracking(self, db_session, lineage_entries):
        """Test lineage tracking capabilities"""
        derived = lineage_entries["derived"]
        source = lineage_entries["source"]
        
        derived.lineage = {
            "source": {
                "id": str(source.id),
                "name": source.name,
                "type": "raw_data"
            },
            "transformations": [{
                "type": "aggregation",
                "timestamp": datetime.utcnow().isoformat(),
                "description": "Daily aggregation"
            }]
        }
        db_session.commit()
        
        retrieved = db_session.get(DataCatalogModel, derived.id)
        assert retrieved.lineage["source"]["id"] == str(source.id)
        assert len(retrieved.lineage["transformations"]) == 1

    def test_downstream_dependencies(self, db_session, lineage_entries):
        """Test downstream dependency tracking"""
        derived = lineage_entries["derived"]
    
        # Add downstream dependency
        downstream = DataCatalogModel(
            name="Downstream Dataset",
            data_lake_id="dl_789",
            meta_info={"level": "campaign", "data_layer": "curated"},
            level_context={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        )
        db_session.add(downstream)
        db_session.commit()
    
        # Use the proper method to add dependency
        derived.add_downstream_dependency(
            str(downstream.id),
            downstream.name,
            "derived_metrics"
        )
        db_session.commit()
    
        # Verify downstream tracking
        retrieved = db_session.get(DataCatalogModel, derived.id)
        assert len(retrieved.lineage["downstream_dependencies"]) == 1
        assert retrieved.lineage["downstream_dependencies"][0]["id"] == str(downstream.id)
        assert retrieved.lineage["downstream_dependencies"][0]["name"] == downstream.name

    # Feedback Tests
    def test_feedback_cycle(self, feedback_processor, sample_catalog_entry, db_session):
        """Test complete feedback loop"""
        db_session.add(sample_catalog_entry)
        db_session.commit()

        # Add feedback
        result = feedback_processor.add_feedback(
            sample_catalog_entry.id,
            "user123",
            "Need better documentation"
        )
        assert result["status"] == "success"

        # Resolve feedback
        feedback_id = sample_catalog_entry.feedback_log[0]["id"]
        resolution = feedback_processor.resolve_feedback(
            sample_catalog_entry.id,
            feedback_id,
            "Added comprehensive documentation"
        )
        assert resolution["status"] == "success"

    def test_feedback_analysis(self, feedback_processor, sample_catalog_entry, db_session):
        """Test feedback analysis and trends"""
        db_session.add(sample_catalog_entry)
        db_session.commit()

        feedback_processor.add_feedback(sample_catalog_entry.id, "user1", "Feedback 1")
        feedback_processor.add_feedback(sample_catalog_entry.id, "user2", "Feedback 2")

        stats = feedback_processor.get_feedback_stats(sample_catalog_entry.id)
        assert stats["total_feedback"] == 2
        
        trends = feedback_processor.analyze_feedback_trends()
        assert "avg_resolution_time" in trends

    # Error Handling Tests
    def test_validation_errors(self, db_session):
        """Test validation error handling"""
        with pytest.raises(ValueError):
            invalid_entry = DataCatalogModel(
                meta_info={"level": "invalid"},
                level_context={"start_date": "2024-01-01"}
            )
            db_session.add(invalid_entry)
            db_session.commit()
import pytest
from datetime import datetime
from app.models.domain.data_catalog_model import DataCatalogModel

class TestLineageTracking:
    """Test suite for data lineage tracking functionality"""
    
    @pytest.fixture(autouse=True)
    def setup(self, db_session):
        """Setup test data"""
        self.db_session = db_session
        
        # Create source catalog entry
        self.source_entry = DataCatalogModel(
            name="Source Dataset",
            data_lake_id="dl_123",
            meta_info={
                "level": "campaign",
                "data_layer": "raw"
            },
            level_context={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        )
        db_session.add(self.source_entry)
        
        # Create derived catalog entry
        self.derived_entry = DataCatalogModel(
            name="Derived Dataset",
            data_lake_id="dl_456",
            meta_info={
                "level": "campaign",
                "data_layer": "processed"
            },
            level_context={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        )
        db_session.add(self.derived_entry)
        db_session.commit()
    
    def test_lineage_emission(self):
        """Test lineage tracking for data transformations"""
        # Update lineage information
        self.derived_entry.lineage = {
            "source": {
                "id": str(self.source_entry.id),
                "name": self.source_entry.name,
                "type": "raw_data"
            },
            "transformations": [
                {
                    "type": "aggregation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "description": "Daily aggregation of metrics"
                }
            ],
            "downstream_dependencies": []
        }
        self.db_session.commit()
        
        # Verify lineage tracking
        updated = self.db_session.get(DataCatalogModel, self.derived_entry.id)
        assert "source" in updated.lineage
        assert updated.lineage["source"]["id"] == str(self.source_entry.id)
        assert len(updated.lineage["transformations"]) == 1
        assert updated.lineage["transformations"][0]["type"] == "aggregation"
    
    def test_lineage_validation(self):
        """Test lineage validation rules"""
        # Test valid lineage update
        self.derived_entry.lineage = {
            "source": {
                "id": str(self.source_entry.id),
                "name": self.source_entry.name,
                "type": "raw_data"
            },
            "transformations": [],
            "downstream_dependencies": []
        }
        self.db_session.commit()
        
        updated = self.db_session.get(DataCatalogModel, self.derived_entry.id)
        assert updated.lineage["source"]["id"] == str(self.source_entry.id)
        
        # Test invalid source reference
        with pytest.raises(ValueError) as exc:
            self.derived_entry.lineage = {
                "source": {
                    "id": "invalid_id",
                    "name": "Invalid Source",
                    "type": "unknown"
                }
            }
            self.db_session.commit()
    
    def test_downstream_tracking(self):
        """Test tracking of downstream dependencies"""
        # Initialize lineage with empty downstream dependencies
        self.derived_entry.lineage = {
            "source": {
                "id": str(self.source_entry.id),
                "name": self.source_entry.name,
                "type": "raw_data"
            },
            "transformations": [],
            "downstream_dependencies": []
        }
        self.db_session.commit()
        
        # Create a downstream dataset
        downstream = DataCatalogModel(
            name="Downstream Dataset",
            data_lake_id="dl_789",
            meta_info={
                "level": "campaign",
                "data_layer": "curated"
            },
            level_context={
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        )
        self.db_session.add(downstream)
        self.db_session.commit()
        
        # Add downstream dependency
        self.derived_entry.add_downstream_dependency(
            str(downstream.id),
            downstream.name,
            "derived_metrics"
        )
        self.db_session.commit()
        
        # Verify downstream tracking
        updated = self.db_session.get(DataCatalogModel, self.derived_entry.id)
        assert len(updated.lineage["downstream_dependencies"]) == 1
        assert updated.lineage["downstream_dependencies"][0]["id"] == str(downstream.id)
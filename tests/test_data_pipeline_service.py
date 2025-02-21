# tests/test_data_pipeline_service.py
import pytest
from sqlalchemy.orm import Session
from app.core.data_lake.data_pipeline_service import DataPipelineService
from app.models.domain.data_catalog_model import DataCatalogModel

class TestDataPipelineService:
    @pytest.fixture(autouse=True)
    def setup(self, db_session: Session, mock_security):
        """Test setup using central fixtures"""
        self.service = DataPipelineService(
            security_manager=mock_security,
            db_session=db_session
        )
        
        # Create test catalog entry
        self.catalog_entry = DataCatalogModel(
            name="Test Dataset",
            data_lake_id="dl_123",
            meta_info={}
        )
        db_session.add(self.catalog_entry)
        db_session.commit()
        self.catalog_id = str(self.catalog_entry.id)

    def test_ingest_data_flow(self, db_session: Session):
        # Test raw ingestion
        test_data = {"sample": "raw_data"}
        self.service.ingest_data(self.catalog_id, test_data, "test_user")
        
        updated = db_session.get(DataCatalogModel, self.catalog_id)
        assert updated.meta_info["layer"] == "raw"
        assert updated.meta_info["raw_data"] == test_data

    def test_full_pipeline_flow(self, db_session: Session):
        # Test full raw -> processed -> curated flow
        test_data = {"sample": "raw_data"}
        
        # Ingest
        self.service.ingest_data(self.catalog_id, test_data, "test_user")
        entry = db_session.get(DataCatalogModel, self.catalog_id)
        assert entry.meta_info["layer"] == "raw"
        
        # Process 
        self.service.process_data(self.catalog_id, "test_user")
        entry = db_session.get(DataCatalogModel, self.catalog_id)
        assert entry.meta_info["layer"] == "processed"
        
        # Curate
        self.service.curate_data(self.catalog_id, "test_user")
        entry = db_session.get(DataCatalogModel, self.catalog_id)
        assert entry.meta_info["layer"] == "curated"
        assert "curated_data" in entry.meta_info
        assert entry.meta_info["processed_data"].get("transformed") is True

    def test_process_without_ingest_fails(self):
        with pytest.raises(ValueError) as exc:
            self.service.process_data(self.catalog_id, "test_user")
        assert "Must be 'raw' first" in str(exc.value)

    def test_curate_without_process_fails(self, db_session: Session):
        self.service.ingest_data(self.catalog_id, {}, "test_user")
        with pytest.raises(ValueError) as exc:
            self.service.curate_data(self.catalog_id, "test_user")
        assert "Must be 'processed' first" in str(exc.value)
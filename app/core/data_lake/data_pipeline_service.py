# /Users/alecposner/WITHIN/app/core/data_lake/data_pipeline_service.py

import logging
from sqlalchemy.orm import Session
from app.core.data_lake.security_manager import SecurityManager
from app.models.domain.data_catalog_model import DataCatalogModel

# For demonstration, define the three layer names:
RAW_LAYER = "raw"
PROCESSED_LAYER = "processed"
CURATED_LAYER = "curated"

class DataPipelineService:
    """
    A layered pipeline service that references DataCatalogModel entries
    to track the 'layer' (raw, processed, curated), usage guidelines, etc.
    """

    def __init__(self, security_manager: SecurityManager, db_session: Session):
        """
        :param security_manager: Manages access checks, encryption, etc.
        :param db_session: SQLAlchemy session for DB ops
        """
        self.security = security_manager
        self.db_session = db_session

    def ingest_data(self, catalog_id: str, data, user_id: str):
        """
        Ingest raw data for a given catalog entry ID. This sets the layer to "raw"
        in the database and optionally stores the data somewhere (S3, local, etc.).
        """
        # 1) Check access
        if not self.security.check_access(user_id, resource=f"catalog:{catalog_id}", action="ingest"):
            raise PermissionError("User not allowed to ingest data.")

        # 2) Load the existing catalog entry
        catalog_entry = self.db_session.query(DataCatalogModel).filter_by(id=catalog_id).first()
        if not catalog_entry:
            raise ValueError(f"Catalog entry {catalog_id} not found.")

        # 3) Store the data somewhere and update meta_info
        #    Example: we might store data in an external location, then update meta_info
        #    For demonstration, let's store it in meta_info
        meta_info = dict(catalog_entry.meta_info or {})  # Make a copy of the dictionary
        meta_info["layer"] = RAW_LAYER
        meta_info["raw_data"] = data
        catalog_entry.meta_info = meta_info  # Assign back to trigger SQLAlchemy update

        self.db_session.commit()

        logging.info(f"[INGEST] Catalog {catalog_id} set to layer: {RAW_LAYER}")

    def process_data(self, catalog_id: str, user_id: str):
        """
        Example of moving from 'raw' to 'processed': 
        e.g., cleaning text, removing stopwords, or normalizing numeric data.
        """
        if not self.security.check_access(user_id, resource=f"catalog:{catalog_id}", action="process"):
            raise PermissionError("User not allowed to process data.")

        catalog_entry = self.db_session.query(DataCatalogModel).filter_by(id=catalog_id).first()
        if not catalog_entry:
            raise ValueError(f"Catalog entry {catalog_id} not found.")

        meta_info = dict(catalog_entry.meta_info or {})  # Make a copy of the dictionary
        current_layer = meta_info.get("layer")
        if current_layer != RAW_LAYER:
            raise ValueError(f"Cannot process data from layer '{current_layer}'. Must be '{RAW_LAYER}' first.")

        # Simulate data transformation
        raw_data = meta_info.get("raw_data", {})
        processed_data = self._transform_data(raw_data)

        # Update DB
        meta_info["layer"] = PROCESSED_LAYER
        meta_info["processed_data"] = processed_data
        catalog_entry.meta_info = meta_info  # Assign back to trigger SQLAlchemy update
        self.db_session.commit()

        logging.info(f"[PROCESS] Catalog {catalog_id} moved from {RAW_LAYER} → {PROCESSED_LAYER}")

    def curate_data(self, catalog_id: str, user_id: str):
        """
        Moves from 'processed' to 'curated', indicating final or validated data.
        """
        if not self.security.check_access(user_id, resource=f"catalog:{catalog_id}", action="curate"):
            raise PermissionError("User not allowed to curate data.")

        catalog_entry = self.db_session.query(DataCatalogModel).filter_by(id=catalog_id).first()
        if not catalog_entry:
            raise ValueError(f"Catalog entry {catalog_id} not found.")

        meta_info = dict(catalog_entry.meta_info or {})  # Make a copy of the dictionary
        current_layer = meta_info.get("layer")
        if current_layer != PROCESSED_LAYER:
            raise ValueError(f"Cannot curate data from layer '{current_layer}'. Must be '{PROCESSED_LAYER}' first.")

        processed_data = meta_info.get("processed_data", {})
        curated_data = self._finalize_data(processed_data)

        # Update DB
        meta_info["layer"] = CURATED_LAYER
        meta_info["curated_data"] = curated_data
        catalog_entry.meta_info = meta_info  # Assign back to trigger SQLAlchemy update
        self.db_session.commit()

        logging.info(f"[CURATE] Catalog {catalog_id} moved from {PROCESSED_LAYER} → {CURATED_LAYER}")

    def _transform_data(self, raw_data):
        """
        Example data transformation logic. For text, you might do:
          - Remove stopwords
          - Tokenize
          - Convert to embeddings
        Or for numeric data, scale or remove outliers, etc.
        """
        # Just a stub:
        if isinstance(raw_data, dict):
            raw_data["transformed"] = True
        return raw_data

    def _finalize_data(self, processed_data):
        """
        Final curation step—maybe topic modeling or summary stats. 
        """
        if isinstance(processed_data, dict):
            processed_data["curated"] = True
        return processed_data

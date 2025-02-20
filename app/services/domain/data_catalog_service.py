from uuid import UUID, uuid4  # Import UUID and uuid4
from app.core.database import SessionLocal
from app.models.domain.data_catalog_model import DataCatalogModel

def add_to_data_catalog(name: str, description: str, data_lake_id: UUID, usage_guidelines: str, metadata: dict):
    db = SessionLocal()
    catalog_entry = DataCatalogModel(
        id=uuid4(),
        name=name,
        description=description,
        data_lake_id=data_lake_id,
        usage_guidelines=usage_guidelines,
        metadata=metadata
    )
    db.add(catalog_entry)
    db.commit()
    db.refresh(catalog_entry)
    return catalog_entry
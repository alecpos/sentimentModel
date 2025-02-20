from app.core.database import SessionLocal
from app.models.domain.data_lake_model import DataLakeModel
from uuid import uuid4

def save_to_data_lake(name: str, data: bytes, metadata: dict):
    db = SessionLocal()
    data_lake_entry = DataLakeModel(
        id=uuid4(),
        name=name,
        data=data,
        metadata=metadata
    )
    db.add(data_lake_entry)
    db.commit()
    db.refresh(data_lake_entry)
    return data_lake_entry
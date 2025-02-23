# /Users/alecposner/WITHIN/app/models/domain/data_catalog_model.py

from datetime import datetime
from uuid import uuid4
from sqlalchemy.orm import validates
from sqlalchemy import (
    Column,
    String,
    JSON,
    DateTime,
    ForeignKey,
    select,
    inspect
)
from sqlalchemy.orm import relationship
from app.core.database import BaseModel

class DataCatalogModel(BaseModel):
    """Model for storing metadata about data assets in the data lake."""

    __tablename__ = "data_catalog"
    level_context = Column(JSON, nullable=False, default=dict)

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(String(1000))

    # Link to 'data_lake.id'
    data_lake_id = Column(String(36), ForeignKey("data_lake.id"), nullable=False)

    usage_guidelines = Column(String(500), nullable=True)
    meta_info = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    schema_definition = Column(JSON, nullable=False, default=dict)
    lineage = Column(JSON, nullable=False, default=dict)
    feedback_log = Column(JSON, nullable=False, default=list)
    version = Column(String(20), default="1.0.0")

    # Relationship back to DataLakeModel
    data_lake_entry = relationship("DataLakeModel", back_populates="catalog_entries")

    LEVEL_CONTEXT_REQUIREMENTS = {
        "campaign": ["start_date", "end_date", "target_demographic"],
        "user": ["user_segments", "age_range", "geography"],
        "environment": ["infra_version", "deployment_region"]
    }
    usage_stats = Column(JSON, nullable=False, default=lambda: {
        "access_count": 0,
        "last_accessed": None,
        "popular_queries": {},
        "access_patterns": {
            "by_hour": {},
            "by_day": {},
            "by_user": {}
        }
    })
    
    def track_usage(self, user_id: str, query_type: str = None):
        """Track dataset usage patterns"""
        now = datetime.utcnow()
        stats = dict(self.usage_stats)
        
        stats["access_count"] += 1
        stats["last_accessed"] = now.isoformat()
        
        hour_key = str(now.hour)
        day_key = now.strftime("%A")
        
        # Update access patterns
        stats["access_patterns"]["by_hour"][hour_key] = \
            stats["access_patterns"]["by_hour"].get(hour_key, 0) + 1
        stats["access_patterns"]["by_day"][day_key] = \
            stats["access_patterns"]["by_day"].get(day_key, 0) + 1
        stats["access_patterns"]["by_user"][user_id] = \
            stats["access_patterns"]["by_user"].get(user_id, 0) + 1
            
        if query_type:
            stats["popular_queries"][query_type] = \
                stats["popular_queries"].get(query_type, 0) + 1
                
        self.usage_stats = stats



    def __init__(self, **kwargs):
        # Validate required fields
        required_fields = ['name', 'data_lake_id']
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Initialize meta_info and level_context before super().__init__
        if 'meta_info' not in kwargs:
            kwargs['meta_info'] = {}
        if 'level_context' not in kwargs:
            kwargs['level_context'] = {}
            
        super().__init__(**kwargs)
        self._validate_level_context_init()
        
        # Initialize feedback_log if None
        if self.feedback_log is None:
            self.feedback_log = []
            
        # Ensure schema_definition exists
        if self.schema_definition is None:
            self.schema_definition = {}
            
        # Initialize lineage with required structure
        if self.lineage is None:
            self.lineage = {}
        if "source" not in self.lineage:
            self.lineage["source"] = {}
        if "transformations" not in self.lineage:
            self.lineage["transformations"] = []
        if "downstream_dependencies" not in self.lineage:
            self.lineage["downstream_dependencies"] = []

    def _validate_level_context_init(self):
        """Validate level-specific context requirements on initialization"""
        level = self.meta_info.get("level")
        if not level:
            return
            
        required_fields = self.LEVEL_CONTEXT_REQUIREMENTS.get(level, [])
        if not all(field in self.level_context for field in required_fields):
            missing = set(required_fields) - set(self.level_context.keys())
            raise ValueError(f"Missing required fields for {level} level: {missing}")

    @validates('level_context')
    def validate_level_context(self, key, context):
        """Validate level context on attribute updates"""
        level = self.meta_info.get("level") if self.meta_info else None
        if level:
            required = self.LEVEL_CONTEXT_REQUIREMENTS.get(level, [])
            if not all(field in context for field in required):
                missing = set(required) - set(context.keys())
                raise ValueError(f"Missing required fields for {level} level: {missing}")
        return context

    def add_feedback(self, user_id: str, comment: str):
        """Add feedback to the catalog entry"""
        feedback_entry = {
            "id": str(uuid4()),
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": False
        }
        
        # Initialize feedback_log if None or not a list
        if not isinstance(self.feedback_log, list):
            self.feedback_log = []
            
        # Create a new list with existing feedback plus new entry to ensure SQLAlchemy detects the change
        new_feedback_log = list(self.feedback_log)
        new_feedback_log.append(feedback_entry)
        self.feedback_log = new_feedback_log
        return feedback_entry

    def resolve_feedback(self, feedback_id: str, resolution_comment: str):
        """Resolve feedback and update version"""
        if not isinstance(self.feedback_log, list):
            self.feedback_log = []
            return None
            
        # Create a new list for feedback to ensure SQLAlchemy detects the change
        updated_feedback = []
        found = False
        
        for feedback in self.feedback_log:
            feedback_copy = dict(feedback)  # Create a copy
            if feedback_copy["id"] == feedback_id:
                feedback_copy["resolved"] = True
                feedback_copy["resolution_comment"] = resolution_comment
                feedback_copy["resolved_at"] = datetime.utcnow().isoformat()
                found = True
                self._increment_version()
            updated_feedback.append(feedback_copy)
        
        self.feedback_log = updated_feedback
        return found

    def _increment_version(self):
        """Increment version number for tracking changes"""
        major, minor, patch = self.version.split(".")
        self.version = f"{major}.{int(minor) + 1}.{patch}"

    @validates('lineage')
    def validate_lineage(self, key, lineage_data):
        """Validate lineage information"""
        if not isinstance(lineage_data, dict):
            raise ValueError("Lineage must be a dictionary")

        # Validate source if present and not empty
        if "source" in lineage_data and lineage_data["source"]:
            source = lineage_data["source"]
            if not isinstance(source, dict):
                raise ValueError("Source must be a dictionary")
            
            required_fields = ["id", "name", "type"]
            if not all(field in source for field in required_fields):
                raise ValueError(f"Source must contain fields: {required_fields}")
            
            # Verify source exists in database using SQLAlchemy 2.0 style
            session = inspect(self).session
            if session:
                stmt = select(DataCatalogModel).filter_by(id=source["id"])
                source_entry = session.scalar(stmt)
                if not source_entry:
                    raise ValueError(f"Source entry with id {source['id']} not found")

        # Ensure transformations is a list if present
        if "transformations" in lineage_data and not isinstance(lineage_data["transformations"], list):
            raise ValueError("Transformations must be a list")

        # Ensure downstream_dependencies is a list if present
        if "downstream_dependencies" in lineage_data and not isinstance(lineage_data["downstream_dependencies"], list):
            raise ValueError("Downstream dependencies must be a list")

        return lineage_data

    def add_downstream_dependency(self, downstream_id: str, downstream_name: str, dependency_type: str):
        """Add a downstream dependency to the lineage"""
        # Ensure lineage is a dictionary
        if not isinstance(self.lineage, dict):
            self.lineage = {}
            
        # Create a deep copy of the current lineage
        current_lineage = {
            "source": dict(self.lineage.get("source", {})) if self.lineage.get("source") else {},
            "transformations": list(self.lineage.get("transformations", [])),
            "downstream_dependencies": list(self.lineage.get("downstream_dependencies", []))
        }
            
        # Add new dependency
        dependency = {
            "id": downstream_id,
            "name": downstream_name,
            "type": dependency_type
        }
        
        # Check if dependency already exists
        if not any(dep.get("id") == downstream_id for dep in current_lineage["downstream_dependencies"]):
            current_lineage["downstream_dependencies"].append(dependency)
        
        # Assign the modified dictionary back to trigger SQLAlchemy change detection
        self.lineage = current_lineage

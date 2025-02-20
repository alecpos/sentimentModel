# app/core/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
import uuid

# Create a base class for declarative models
Base = declarative_base()

# Database connection configuration
DATABASE_URL = "sqlite:///./test.db"  # Update with your actual database URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL, 
    echo=False,  # Set to True for SQL logging during development
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Base model with common functionality
class BaseModel(Base):
    """Base model with common attributes and methods"""
    __abstract__ = True

    @classmethod
    def generate_uuid(cls):
        """Generate a UUID string"""
        return str(uuid.uuid4())

    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

# Create all tables defined in models
def create_tables():
    """Create all database tables defined in models"""
    Base.metadata.create_all(bind=engine)
# app/core/db/base.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

# Create base class for declarative models
Base = declarative_base()

# Database connection configuration
DATABASE_URL = "sqlite:///./test.db"  # Update with your actual database URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL, 
    echo=False,
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

def create_tables():
    """Create all database tables defined in models"""
    Base.metadata.create_all(bind=engine)
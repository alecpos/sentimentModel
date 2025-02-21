from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/app_db"
    TEST_DATABASE_URL: str = "sqlite:///./test.db"
    
    # Elasticsearch
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    
    model_config = ConfigDict(env_file=".env")

settings = Settings()
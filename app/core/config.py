"""
Core configuration settings
"""

import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    PROJECT_NAME: str = "eDiscovery LLM Retrieval System"
    PROJECT_DESCRIPTION: str = "Production-ready document processing system with QA, Summarization, Classification, and NER capabilities"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # OpenSearch Configuration
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USE_SSL: bool = False
    OPENSEARCH_VERIFY_CERTS: bool = False
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    
    # BERT NER Configuration
    BERT_MODEL_NAME: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    BERT_DEVICE: str = "auto"  # auto, cpu, cuda
    
    # Task Queue Configuration (for future Celery integration)
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # Data paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    CACHE_DIR: str = "cache"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

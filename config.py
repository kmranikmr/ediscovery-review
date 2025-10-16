"""
Configuration settings for the LLM retrieval system.
This module contains shared configuration values to avoid circular imports.
"""
import os

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # 'ollama', 'openai', 'huggingface'
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# Derived constants
OLLAMA_API_BASE = LLM_BASE_URL
OLLAMA_MODEL = LLM_MODEL_NAME

# Debug settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
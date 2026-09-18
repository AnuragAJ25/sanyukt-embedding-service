from typing import List, Optional
import os
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Microservice application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    EMBEDDING_MODEL: str = Field(
        default="intfloat/multilingual-e5-small",
        description="HuggingFace model identifier for sentence-transformers",
    )
    EMBEDDING_MODEL_REVISION: str = Field(
        default="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        description="Pinned HuggingFace model commit revision hash",
    )
    EMBEDDING_API_TOKEN: str = Field(
        default="",
        description="Secret Bearer token required to access embedding endpoints",
    )
    MAX_BATCH_SIZE: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Maximum allowed texts in a single batch request",
    )
    MAX_CONCURRENT_INFERENCE: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Maximum concurrent inference operations allowed (CPU default: 1)",
    )
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for Uvicorn server",
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Host interface to bind Uvicorn server",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    DEVICE: Optional[str] = Field(
        default=None,
        description="Inference device: 'cpu' or 'cuda'. If None, auto-detected.",
    )
    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=list,
        description="CORS allowed origins. Keep empty for backend-only usage.",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=384,
        description="Expected embedding dimension for multilingual-e5-small",
    )

    def is_token_configured(self) -> bool:
        """Check whether a non-empty EMBEDDING_API_TOKEN is configured."""
        return bool(self.EMBEDDING_API_TOKEN and self.EMBEDDING_API_TOKEN.strip())

    def get_device(self) -> str:
        """Resolve device to 'cuda' or 'cpu'."""
        if self.DEVICE:
            return self.DEVICE.lower()
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()

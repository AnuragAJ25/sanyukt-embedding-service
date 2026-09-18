from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    """Supported E5 input types determining automatic prefix handling."""
    QUERY = "query"
    PASSAGE = "passage"


class SingleEmbeddingRequest(BaseModel):
    """Payload for generating an embedding for a single text."""
    text: str = Field(
        ...,
        description="The raw text string to embed.",
        min_length=1,
        max_length=8192,
        examples=["tractor subsidy"],
    )
    input_type: InputType = Field(
        ...,
        description="Specifies whether text is a search query or an indexed passage.",
        examples=[InputType.QUERY],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("text cannot be empty or whitespace-only")
        return stripped


class SingleEmbeddingResponse(BaseModel):
    """Response containing the 384-dimensional normalized embedding."""
    model: str = Field(
        ...,
        description="Name of the model used to produce the embedding.",
        examples=["intfloat/multilingual-e5-small"],
    )
    revision: Optional[str] = Field(
        default=None,
        description="Pinned model commit revision hash.",
        examples=["614241f622f53c4eeff9890bdc4f31cfecc418b3"],
    )
    dimensions: int = Field(
        ...,
        description="Dimension size of the embedding vector.",
        examples=[384],
    )
    embedding: List[float] = Field(
        ...,
        description="384-dimensional unit-normalized float array.",
    )


class BatchEmbeddingRequest(BaseModel):
    """Payload for generating embeddings for a batch of texts."""
    texts: List[str] = Field(
        ...,
        description="Array of texts to embed.",
        min_length=1,
        examples=[[
            "Age Nationality and Domicile Certificate",
            "Sub-Mission on Farm Mechanization",
            "Skill Development Training",
        ]],
    )
    input_type: InputType = Field(
        ...,
        description="Specifies whether texts are search queries or indexed passages.",
        examples=[InputType.PASSAGE],
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("texts list cannot be empty")
        cleaned: List[str] = []
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"item at index {i} must be a string")
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"item at index {i} cannot be empty or whitespace-only")
            if len(stripped) > 8192:
                raise ValueError(f"item at index {i} exceeds maximum character limit of 8192")
            cleaned.append(stripped)
        return cleaned


class BatchEmbeddingResponse(BaseModel):
    """Response containing batch embeddings."""
    model: str = Field(
        ...,
        description="Name of the model used to produce the embeddings.",
        examples=["intfloat/multilingual-e5-small"],
    )
    revision: Optional[str] = Field(
        default=None,
        description="Pinned model commit revision hash.",
        examples=["614241f622f53c4eeff9890bdc4f31cfecc418b3"],
    )
    dimensions: int = Field(
        ...,
        description="Dimension size of each embedding vector.",
        examples=[384],
    )
    count: int = Field(
        ...,
        description="Number of embeddings generated.",
        examples=[3],
    )
    embeddings: List[List[float]] = Field(
        ...,
        description="List of 384-dimensional unit-normalized float arrays.",
    )


class HealthResponse(BaseModel):
    """Service health and model readiness status."""
    status: str = Field(default="ok", examples=["ok"])
    model: str = Field(default="intfloat/multilingual-e5-small", examples=["intfloat/multilingual-e5-small"])
    revision: Optional[str] = Field(
        default=None,
        description="Pinned model commit revision hash.",
        examples=["614241f622f53c4eeff9890bdc4f31cfecc418b3"],
    )
    dimensions: int = Field(default=384, examples=[384])
    ready: bool = Field(default=True, examples=[True])
    device: str = Field(default="cpu", examples=["cpu"])

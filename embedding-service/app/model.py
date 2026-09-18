import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import InputType

logger = logging.getLogger("embedding-service.model")


class EmbeddingEngine:
    """Singleton-style wrapper for loading and executing sentence-transformers model."""

    def __init__(self) -> None:
        self.model: Optional[SentenceTransformer] = None
        self.is_ready: bool = False
        self.model_name: str = settings.EMBEDDING_MODEL
        self.model_revision: str = settings.EMBEDDING_MODEL_REVISION
        self.device: str = settings.get_device()
        self.dimensions: int = settings.EMBEDDING_DIMENSIONS

    def load(self) -> None:
        """Loads model once into memory and performs startup inference warmup."""
        logger.info(
            f"Loading embedding model '{self.model_name}' (revision: '{self.model_revision}') "
            f"on device '{self.device}'..."
        )
        try:
            self.model = SentenceTransformer(
                self.model_name,
                revision=self.model_revision,
                device=self.device,
            )
            # Perform a lightweight inference warmup check
            warmup_text = ["passage: system startup verification"]
            warmup_embedding = self.model.encode(
                warmup_text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if warmup_embedding is None or len(warmup_embedding) == 0:
                raise RuntimeError("Warmup inference produced empty embedding.")

            actual_dim = len(warmup_embedding[0])
            if actual_dim != self.dimensions:
                raise ValueError(
                    f"Model embedding dimension mismatch: expected {self.dimensions}, got {actual_dim}"
                )

            self.is_ready = True
            logger.info(
                f"Model successfully loaded and warmed up. Dimensions: {self.dimensions}, Device: {self.device}"
            )
        except Exception as exc:
            self.is_ready = False
            self.model = None
            logger.error(f"Failed to load embedding model '{self.model_name}': {exc}", exc_info=True)
            raise

    def embed_texts(self, texts: List[str], input_type: InputType) -> List[List[float]]:
        """
        Embeds a list of texts with automatic E5 prefixing and L2 normalization.

        Prefix handling:
        - query: "query: <text>"
        - passage: "passage: <text>"
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("Embedding engine is not initialized or model is not loaded.")

        if not texts:
            return []

        # Automatically format with E5 required prefix
        prefix = "query: " if input_type == InputType.QUERY else "passage: "
        prefixed_texts = [f"{prefix}{text}" for text in texts]

        try:
            # Generate unit-normalized embeddings
            embeddings = self.model.encode(
                prefixed_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Ensure pure Python list of floats (no raw numpy objects)
            if isinstance(embeddings, np.ndarray):
                embeddings_list: List[List[float]] = embeddings.tolist()
            else:
                embeddings_list = [list(vec) for vec in embeddings]

            # Enforce dimension check
            for vec in embeddings_list:
                if len(vec) != self.dimensions:
                    raise ValueError(
                        f"Generated vector has incorrect dimension: {len(vec)} != {self.dimensions}"
                    )

            return embeddings_list

        except Exception as exc:
            # Privacy note: Never log raw input text
            logger.error(
                f"Embedding generation error for batch_size={len(texts)}, input_type={input_type}: {exc}"
            )
            raise


# Global single instance across the process lifecycle
engine = EmbeddingEngine()

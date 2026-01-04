#!/usr/bin/env python3
"""Content-based feature extraction using sentence embeddings.

Uses sentence-transformers to create semantic embeddings from email text.
This captures the meaning of email content beyond keyword-based features.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# Default model: all-mpnet-base-v2
# - 768-dim embeddings
# - High quality semantic representations
# - Better disambiguation of similar texts
DEFAULT_MODEL = 'all-mpnet-base-v2'
DEFAULT_EMBEDDING_DIM = 768

# Large model: all-mpnet-base-v2
# - 768-dim embeddings
# - Slower inference but better semantic quality
# - Recommended for ensemble's large_embed variant
LARGE_MODEL = 'all-mpnet-base-v2'
LARGE_EMBEDDING_DIM = 768

# Model name to embedding dimension mapping
MODEL_EMBEDDING_DIMS = {
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'paraphrase-MiniLM-L6-v2': 384,
    'paraphrase-mpnet-base-v2': 768,
}

# Large model: all-mpnet-base-v2
# - 768-dim embeddings
# - Slower inference but better semantic quality
# - Recommended for ensemble's large_embed variant
LARGE_MODEL = 'all-mpnet-base-v2'
LARGE_EMBEDDING_DIM = 768

# Model name to embedding dimension mapping
MODEL_EMBEDDING_DIMS = {
    'all-MiniLM-L6-v2': 384,
    'all-mpnet-base-v2': 768,
    'paraphrase-MiniLM-L6-v2': 384,
    'paraphrase-mpnet-base-v2': 768,
}

# Maximum text length to process (chars)
MAX_SUBJECT_LEN = 256
MAX_BODY_LEN = 2048


@dataclass
class ContentFeatures:
    """Content-based features from email text embeddings."""
    # Raw embeddings
    subject_embedding: Union["np.ndarray", list[float]] = field(default_factory=lambda: [0.0] * DEFAULT_EMBEDDING_DIM)
    body_embedding: Union["np.ndarray", list[float]] = field(default_factory=lambda: [0.0] * DEFAULT_EMBEDDING_DIM)

    # Combined embedding (weighted average of subject and body)
    combined_embedding: Union["np.ndarray", list[float]] = field(default_factory=lambda: [0.0] * DEFAULT_EMBEDDING_DIM)

    # Text statistics
    subject_length: int = 0
    body_length: int = 0
    body_word_count: int = 0

    # Embedding dimension
    embedding_dim: int = DEFAULT_EMBEDDING_DIM

    def to_feature_vector(self) -> Union["np.ndarray", list[float]]:
        """Convert to feature vector for ML pipeline.

        Returns only the combined embedding to keep dimensions manageable.
        Total: embedding_dim (default 768)
        """
        if HAS_NUMPY:
            return np.asarray(self.combined_embedding, dtype=np.float32)
        return list(self.combined_embedding)

    @property
    def feature_dim(self) -> int:
        """Return feature vector dimensionality."""
        return self.embedding_dim


class ContentFeatureExtractor:
    """Extracts content features using sentence embeddings.

    Lazily loads the model on first use to avoid startup overhead.
    Thread-safe for inference (model is loaded once).

    Example:
        >>> extractor = ContentFeatureExtractor()
        >>> features = extractor.extract(email)
        >>> embedding = features.to_feature_vector()  # 768-dim vector
    """

    _model: Optional["SentenceTransformer"] = None
    _model_name: str = DEFAULT_MODEL
    _embedding_dim: int = DEFAULT_EMBEDDING_DIM

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize content feature extractor.

        Args:
            model_name: Name of sentence-transformers model to use
            device: Device to run model on ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for encoding multiple texts
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazily load and return the sentence transformer model."""
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required for content features. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded, embedding dim: {self._embedding_dim}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        # Return expected dim from known models without loading
        if self.model_name in MODEL_EMBEDDING_DIMS:
            return MODEL_EMBEDDING_DIMS[self.model_name]
        # Have to load model to know dimension
        return self.model.get_sentence_embedding_dimension()

    def _preprocess_text(self, text: str, max_len: int) -> str:
        """Clean and truncate text for embedding."""
        if not text:
            return ""
        # Basic cleanup
        text = text.strip()
        # Replace multiple whitespace with single space
        text = ' '.join(text.split())
        # Truncate
        if len(text) > max_len:
            text = text[:max_len]
        return text

    def _encode(self, texts: list[str]) -> "np.ndarray":
        """Encode a list of texts to embeddings."""
        # Filter empty strings but keep track of positions
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        non_empty_texts = [t for t in texts if t]

        if not non_empty_texts:
            # All empty - return zeros
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        # Encode non-empty texts
        embeddings = self.model.encode(
            non_empty_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Build result array with zeros for empty positions
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for idx, emb_idx in enumerate(non_empty_indices):
            result[emb_idx] = embeddings[idx]

        return result

    def extract(self, email: dict) -> ContentFeatures:
        """Extract content features from a single email.

        Args:
            email: Email dict with 'subject' and 'body' fields

        Returns:
            ContentFeatures with embeddings
        """
        subject = self._preprocess_text(email.get('subject', ''), MAX_SUBJECT_LEN)
        body = self._preprocess_text(email.get('body', ''), MAX_BODY_LEN)

        # Encode both texts in one batch for efficiency
        embeddings = self._encode([subject, body])
        subject_emb = embeddings[0]
        body_emb = embeddings[1]

        # Combine embeddings with weights
        # Subject is often more informative per character, weight it higher
        subject_weight = 0.4
        body_weight = 0.6

        # Adjust weights if one is empty
        if not subject:
            subject_weight = 0.0
            body_weight = 1.0
        elif not body:
            subject_weight = 1.0
            body_weight = 0.0

        combined = subject_weight * subject_emb + body_weight * body_emb

        # Normalize combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return ContentFeatures(
            subject_embedding=subject_emb,
            body_embedding=body_emb,
            combined_embedding=combined,
            subject_length=len(subject),
            body_length=len(body),
            body_word_count=len(body.split()) if body else 0,
            embedding_dim=self.embedding_dim,
        )

    def extract_batch(self, emails: list[dict]) -> list[ContentFeatures]:
        """Extract content features from multiple emails efficiently.

        Batches all texts together for GPU efficiency.

        Args:
            emails: List of email dicts

        Returns:
            List of ContentFeatures
        """
        # Collect all texts
        subjects = [self._preprocess_text(e.get('subject', ''), MAX_SUBJECT_LEN) for e in emails]
        bodies = [self._preprocess_text(e.get('body', ''), MAX_BODY_LEN) for e in emails]

        # Encode all at once
        all_texts = subjects + bodies
        all_embeddings = self._encode(all_texts)

        n = len(emails)
        subject_embeddings = all_embeddings[:n]
        body_embeddings = all_embeddings[n:]

        # Build features
        features = []
        for i, email in enumerate(emails):
            subject = subjects[i]
            body = bodies[i]
            subject_emb = subject_embeddings[i]
            body_emb = body_embeddings[i]

            # Combine
            subject_weight = 0.4
            body_weight = 0.6
            if not subject:
                subject_weight = 0.0
                body_weight = 1.0
            elif not body:
                subject_weight = 1.0
                body_weight = 0.0

            combined = subject_weight * subject_emb + body_weight * body_emb
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            features.append(ContentFeatures(
                subject_embedding=subject_emb,
                body_embedding=body_emb,
                combined_embedding=combined,
                subject_length=len(subject),
                body_length=len(body),
                body_word_count=len(body.split()) if body else 0,
                embedding_dim=self.embedding_dim,
            ))

        return features


# Global extractor instance for convenience (lazy loaded)
_global_extractor: Optional[ContentFeatureExtractor] = None


def get_content_extractor(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> ContentFeatureExtractor:
    """Get or create global content feature extractor.

    Uses lazy loading to avoid startup overhead when not needed.
    """
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = ContentFeatureExtractor(model_name=model_name, device=device)
    return _global_extractor


def extract_content_features(email: dict) -> ContentFeatures:
    """Extract content features from an email using the global extractor."""
    return get_content_extractor().extract(email)


def get_embedding_dim() -> int:
    """Return the embedding dimension without loading the model."""
    return DEFAULT_EMBEDDING_DIM


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("CONTENT FEATURE EXTRACTION TEST")
    print("=" * 60)

    sample_email = {
        'subject': 'URGENT: Project Eagle Phase II - Review Required by Friday',
        'body': """
        Hi Jane,

        I need your decision on the Project Eagle Phase II proposal by end of day Friday.

        Please review the attached document and let me know if you have any concerns.
        This is time-sensitive as we need to submit to the board by Monday.

        Key items for your review:
        - Budget estimates (Contract #12345)
        - Timeline for Q2 deliverables
        - Resource allocation for the team

        Can you also send me the updated risk assessment? We need it before we can
        proceed with the client meeting next week.

        Thanks,
        John
        """,
    }

    print("\nLoading model...")
    extractor = ContentFeatureExtractor()

    print("\nExtracting features...")
    features = extractor.extract(sample_email)

    print(f"\nEmbedding dimension: {features.embedding_dim}")
    print(f"Subject length: {features.subject_length} chars")
    print(f"Body length: {features.body_length} chars")
    print(f"Body word count: {features.body_word_count}")

    vec = features.to_feature_vector()
    print(f"\nFeature vector shape: {vec.shape}")
    print(f"Feature vector dtype: {vec.dtype}")
    print(f"First 10 values: {vec[:10]}")
    print(f"Vector norm: {np.linalg.norm(vec):.4f}")

    # Test batch extraction
    print("\nTesting batch extraction...")
    emails = [sample_email] * 5
    batch_features = extractor.extract_batch(emails)
    print(f"Batch size: {len(batch_features)}")

    # Compare single vs batch (should be identical)
    single_vec = features.to_feature_vector()
    batch_vec = batch_features[0].to_feature_vector()
    diff = np.abs(single_vec - batch_vec).max()
    print(f"Max diff between single and batch: {diff:.8f}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

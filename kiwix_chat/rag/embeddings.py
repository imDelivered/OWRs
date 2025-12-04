"""Embedding generation for RAG system using sentence-transformers."""

import sys
from typing import List, Optional

# Lazy-loaded model
_embedding_model = None
# Using BGE (BAAI General Embedding) model
# Default: English model, but works reasonably well with other languages
# For better multilingual support, consider: "BAAI/bge-m3" (multilingual, ~420MB)
# Options: 
#   - "BAAI/bge-base-en-v1.5" (English, ~110MB) - default, works with any ZIM
#   - "BAAI/bge-large-en-v1.5" (English, ~1.3GB) - better quality
#   - "BAAI/bge-m3" (Multilingual, ~420MB) - best for non-English ZIM files
_embedding_model_name = "BAAI/bge-base-en-v1.5"


def initialize_embedding_model():
    """Lazy load the embedding model on first use."""
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[rag] Loading embedding model: {_embedding_model_name}", file=sys.stderr)
        print(f"[rag] First-time download may take a few minutes (~110MB)...", file=sys.stderr)
        
        # BGE models need instruction prefix for queries
        try:
            _embedding_model = SentenceTransformer(_embedding_model_name)
            dim = _embedding_model.get_sentence_embedding_dimension()
            print(f"[rag] Embedding model loaded successfully (dimension: {dim})", file=sys.stderr)
            return _embedding_model
        except OSError as e:
            if "No such file or directory" in str(e) or "model" in str(e).lower():
                print(f"[rag] ERROR: Model file not found. This may be a download issue.", file=sys.stderr)
                print(f"[rag] Try: python3 -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"{_embedding_model_name}\")'", file=sys.stderr)
            raise
        except Exception as e:
            print(f"[rag] ERROR: Failed to load embedding model: {e}", file=sys.stderr)
            print(f"[rag] Check your internet connection (first download) or model cache.", file=sys.stderr)
            raise
    except ImportError:
        print("[rag] ERROR: sentence-transformers not installed.", file=sys.stderr)
        print("[rag] Install with: pip install sentence-transformers", file=sys.stderr)
        raise


def generate_embedding(text: str, is_query: bool = True) -> List[float]:
    """Generate embedding for a single text.
    
    BGE models require instruction prefixes for optimal performance:
    - Queries: "Represent this sentence for searching relevant passages:"
    - Passages: "Represent this sentence for retrieving relevant passages:"
    
    Args:
        text: Text to embed
        is_query: Whether this is a query (True) or a passage/chunk (False)
        
    Returns:
        List of floats representing the embedding vector
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        model = initialize_embedding_model()
        return [0.0] * model.get_sentence_embedding_dimension()
    
    model = initialize_embedding_model()
    
    # BGE models work better with instruction prefixes
    if is_query:
        # For queries, add instruction prefix (BGE best practice)
        instruction = "Represent this sentence for searching relevant passages:"
        text_with_instruction = f"{instruction} {text}"
    else:
        # For passages/chunks, use as-is (or add passage instruction if needed)
        text_with_instruction = text
    
    embedding = model.encode(text_with_instruction, convert_to_numpy=True, show_progress_bar=False)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str], batch_size: int = 32, show_progress: bool = False, is_query: bool = False) -> List[List[float]]:
    """Generate embeddings for multiple texts in batch.
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for processing
        show_progress: Whether to show progress bar
        is_query: Whether these are queries (True) or passages/chunks (False)
        
    Returns:
        List of embedding vectors (one per input text)
    """
    if not texts:
        return []
    
    # Filter out empty texts
    non_empty_texts = [text if text and text.strip() else " " for text in texts]
    
    # BGE models work better with instruction prefixes for queries
    if is_query:
        instruction = "Represent this sentence for searching relevant passages:"
        non_empty_texts = [f"{instruction} {text}" for text in non_empty_texts]
    
    model = initialize_embedding_model()
    embeddings = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings produced by the model."""
    model = initialize_embedding_model()
    return model.get_sentence_embedding_dimension()


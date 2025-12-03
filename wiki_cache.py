#!/usr/bin/env python3
"""
Caching layer for Kiwix RAG system.
Provides LRU caches for articles, topic extraction, and search results.
"""

from typing import Optional, List, Tuple, Dict, Any
from functools import lru_cache
import hashlib
import re


# In-memory caches (will be initialized on first use)
_article_cache: Dict[str, Tuple[Any, float]] = {}  # key: article_title, value: (result, timestamp)
_topic_extraction_cache: Dict[str, List[str]] = {}  # key: normalized_query, value: topics
_search_cache: Dict[str, Optional[str]] = {}  # key: search_query, value: href

# Cache configuration
ARTICLE_CACHE_MAX_SIZE = 1000
TOPIC_CACHE_MAX_SIZE = 500
SEARCH_CACHE_MAX_SIZE = 2000
CACHE_TTL_SECONDS = 3600  # 1 hour TTL


def _normalize_query(query: str) -> str:
    """Normalize query for caching (lowercase, remove extra whitespace)."""
    return re.sub(r'\s+', ' ', query.lower().strip())


def _get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    normalized = _normalize_query(query)
    return hashlib.md5(normalized.encode()).hexdigest()


def get_cached_article(article_title: str) -> Optional[Any]:
    """Get cached article if available.
    
    Args:
        article_title: Title of the article
        
    Returns:
        Cached article result (text, links, href) or None if not cached
    """
    import time
    key = article_title.lower().strip()
    
    if key in _article_cache:
        result, timestamp = _article_cache[key]
        # Check TTL
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return result
        else:
            # Expired, remove from cache
            del _article_cache[key]
    
    return None


def cache_article(article_title: str, result: Any) -> None:
    """Cache an article result.
    
    Args:
        article_title: Title of the article
        result: Article result (text, links, href)
    """
    import time
    key = article_title.lower().strip()
    
    # Implement LRU: remove oldest if cache is full
    if len(_article_cache) >= ARTICLE_CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_article_cache))
        del _article_cache[oldest_key]
    
    _article_cache[key] = (result, time.time())


def get_cached_topics(query: str) -> Optional[List[str]]:
    """Get cached topic extraction result if available.
    
    Args:
        query: User query
        
    Returns:
        Cached topics list or None if not cached
    """
    cache_key = _get_cache_key(query)
    return _topic_extraction_cache.get(cache_key)


def cache_topics(query: str, topics: List[str]) -> None:
    """Cache topic extraction result.
    
    Args:
        query: User query
        topics: Extracted topics list
    """
    cache_key = _get_cache_key(query)
    
    # Implement LRU: remove oldest if cache is full
    if len(_topic_extraction_cache) >= TOPIC_CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_topic_extraction_cache))
        del _topic_extraction_cache[oldest_key]
    
    _topic_extraction_cache[cache_key] = topics


def get_cached_search(query: str) -> Optional[str]:
    """Get cached search result if available.
    
    Args:
        query: Search query
        
    Returns:
        Cached href or None if not cached
    """
    cache_key = _get_cache_key(query)
    return _search_cache.get(cache_key)


def cache_search(query: str, href: Optional[str]) -> None:
    """Cache search result.
    
    Args:
        query: Search query
        href: Article href (or None if not found)
    """
    cache_key = _get_cache_key(query)
    
    # Implement LRU: remove oldest if cache is full
    if len(_search_cache) >= SEARCH_CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_search_cache))
        del _search_cache[oldest_key]
    
    _search_cache[cache_key] = href


def clear_all_caches() -> None:
    """Clear all caches."""
    _article_cache.clear()
    _topic_extraction_cache.clear()
    _search_cache.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics.
    
    Returns:
        Dict with cache sizes
    """
    return {
        'article_cache_size': len(_article_cache),
        'topic_cache_size': len(_topic_extraction_cache),
        'search_cache_size': len(_search_cache),
        'article_cache_max': ARTICLE_CACHE_MAX_SIZE,
        'topic_cache_max': TOPIC_CACHE_MAX_SIZE,
        'search_cache_max': SEARCH_CACHE_MAX_SIZE,
    }



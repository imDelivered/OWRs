"""
Multi-Joint RAG System - Reasoning Joints for Improved Retrieval

This module implements three reasoning joints that use small LLMs to guide
the retrieval process and prevent hallucinations:

1. EntityExtractorJoint - Extracts entities and aliases from queries
2. ArticleScorerJoint - Scores article relevance to entities
3. ChunkFilterJoint - Filters chunks by query relevance
"""

import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from chatbot import config


def debug_print(joint_name: str, msg: str):
    """Print debug message for a specific joint."""
    if config.DEBUG:
        print(f"[DEBUG:{joint_name}] {msg}", file=sys.stderr)


def ollama_call(model: str, prompt: str, temperature: float = 0.0, timeout: int = 5) -> str:
    """
    Call Ollama API with a prompt and return response.
    
    Args:
        model: Ollama model name
        prompt: Prompt text
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        
    Returns:
        Response text from model
    """
    url = config.OLLAMA_CHAT_URL
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature}
    }).encode("utf-8")
    
    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            obj = json.loads(data.decode("utf-8"))
            if obj.get("error"):
                raise RuntimeError(str(obj["error"]))
            message = obj.get("message", {})
            return message.get("content", "")
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama: {e.reason}") from e


class EntityExtractorJoint:
    """
    Joint 1: Entity Extraction
    
    Extracts the main entity, type, action, and aliases from a user query.
    Uses llama3.2:1b for fast, focused entity recognition.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.ENTITY_JOINT_MODEL
        self.temperature = config.ENTITY_JOINT_TEMP
        debug_print("JOINT1:INIT", f"EntityExtractor initialized with {self.model}")
    
    def extract(self, query: str) -> Dict[str, any]:
        """
        Extract entity information from query.
        
        Args:
            query: User query string
            
        Returns:
            Dict with keys: entity, entity_type, action, aliases
        """
        debug_print("JOINT1:ENTITY", f"Extracting entities from: '{query}'")
        start_time = time.time()
        
        prompt = f"""You are an entity extraction system. Extract the main entity from this query.

Query: {query}

Return ONLY valid JSON with this exact structure:
{{
  "entity": "full entity name",
  "entity_type": "person|place|event|concept",
  "action": "what question/action about entity",
  "aliases": ["alternate names or spellings"]
}}

Do not include any examples. Return ONLY the JSON object.
"""

        try:
            response = ollama_call(self.model, prompt, self.temperature, config.JOINT_TIMEOUT)
            debug_print("JOINT1:ENTITY", f"Raw response: {response[:200]}...")
            
            # Extract JSON from response (handle cases where model adds text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            # Validate result structure
            required_keys = ['entity', 'entity_type', 'action', 'aliases']
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required keys. Got: {result.keys()}")
            
            elapsed = time.time() - start_time
            debug_print("JOINT1:ENTITY", f"Extracted: entity='{result['entity']}', type={result['entity_type']}, action={result['action']}")
            debug_print("JOINT1:ENTITY", f"Aliases: {result['aliases']}")
            debug_print("JOINT1:ENTITY", f"Extraction took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            debug_print("JOINT1:ENTITY", f"Extraction failed: {type(e).__name__}: {e}")
            # Fallback: return query as entity
            debug_print("JOINT1:ENTITY", "Using fallback: query as entity")
            return {
                "entity": query,
                "entity_type": "unknown",
                "action": "information",
                "aliases": []
            }


class ArticleScorerJoint:
    """
    Joint 2: Article Scoring
    
    Scores Wikipedia article titles by relevance to the extracted entity.
    Uses qwen2.5:0.5b for fast scoring.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.SCORER_JOINT_MODEL
        self.temperature = config.SCORER_JOINT_TEMP
        debug_print("JOINT2:INIT", f"ArticleScorer initialized with {self.model}")
    
    def score(self, entity_info: Dict, article_titles: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Score article titles by relevance to entity.
        
        Args:
            entity_info: Entity information from EntityExtractorJoint
            article_titles: List of Wikipedia article titles
            top_k: Return top K scored articles
            
        Returns:
            List of (title, score) tuples, sorted by score descending
        """
        if not article_titles:
            debug_print("JOINT2:SCORER", "No articles to score")
            return []
        
        debug_print("JOINT2:SCORER", f"Scoring {len(article_titles)} articles for entity '{entity_info['entity']}'")
        start_time = time.time()
        
        # Format article titles for prompt (limit to prevent token overflow)
        articles_formatted = "\n".join([f"{i+1}. {title}" for i, title in enumerate(article_titles[:20])])
        
        prompt = f"""Score how relevant these Wikipedia articles are to answering a question about this entity.

Entity: {entity_info['entity']}
Type: {entity_info['entity_type']}
Question about: {entity_info['action']}
Also known as: {', '.join(entity_info['aliases'])}

Articles:
{articles_formatted}

Rate each article 0-10 where:
- 10 = Perfect match, exactly what we need
- 7-9 = Highly relevant
- 4-6 = Somewhat relevant
- 1-3 = Barely relevant  
- 0 = Not relevant

Return ONLY a JSON array in this exact format:
[
  {{"title": "Article Name", "score": 10}},
  {{"title": "Another Article", "score": 5}}
]

Return ALL articles with scores. No explanation, only JSON."""

        try:
            response = ollama_call(self.model, prompt, self.temperature, config.JOINT_TIMEOUT)
            debug_print("JOINT2:SCORER", f"Raw response: {response[:200]}...")
            
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                scores = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            # Convert to list of tuples and sort
            scored_articles = [(item['title'], float(item['score'])) for item in scores]
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            
            elapsed = time.time() - start_time
            debug_print("JOINT2:SCORER", f"Scored {len(scored_articles)} articles in {elapsed:.2f}s")
            debug_print("JOINT2:SCORER", f"Top 5 scores: {scored_articles[:5]}")
            
            return scored_articles[:top_k]
            
        except Exception as e:
            debug_print("JOINT2:SCORER", f"Scoring failed: {type(e).__name__}: {e}")
            # Fallback: return all articles with equal scores
            debug_print("JOINT2:SCORER", "Using fallback: equal scores")
            return [(title, 5.0) for title in article_titles[:top_k]]


class ChunkFilterJoint:
    """
    Joint 3: Chunk Filtering
    
    Filters retrieved chunks by relevance to the original query.
    Uses llama3.2:1b for intelligent chunk evaluation.
    """
    
    def __init__(self, model: str = None):
        self.model = model or config.FILTER_JOINT_MODEL
        self.temperature = config.FILTER_JOINT_TEMP
        debug_print("JOINT3:INIT", f"ChunkFilter initialized with {self.model}")
    
    def filter(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Filter chunks by query relevance.
        
        Args:
            query: Original user query
            chunks: List of chunk dicts with 'text' and 'metadata' keys
            top_k: Return top K relevant chunks
            
        Returns:
            List of chunk dicts, sorted by relevance
        """
        if not chunks:
            debug_print("JOINT3:FILTER", "No chunks to filter")
            return []
        
        debug_print("JOINT3:FILTER", f"Filtering {len(chunks)} chunks for query '{query}'")
        start_time = time.time()
        
        # Format chunks for prompt (truncate long chunks, limit to 20)
        chunks_formatted = []
        for i, chunk in enumerate(chunks[:20]):
            text = chunk['text'][:300]  # Truncate to 300 chars
            chunks_formatted.append(f"{i+1}. {text}...")
        
        chunks_text = "\n\n".join(chunks_formatted)
        
        prompt = f"""Rate these text chunks for how well they answer this query.

Query: {query}

Chunks:
{chunks_text}

Rate each chunk 0-10 where:
- 10 = Directly answers the query
- 7-9 = Highly relevant context
- 4-6 = Related information
- 1-3 = Tangentially related
- 0 = Not relevant

Return ONLY a JSON array:
[
  {{"chunk_id": 1, "score": 10}},
  {{"chunk_id": 2, "score": 3}}
]

Rate ALL chunks. No explanation, only JSON."""

        try:
            response = ollama_call(self.model, prompt, self.temperature, config.JOINT_TIMEOUT)
            debug_print("JOINT3:FILTER", f"Raw response: {response[:200]}...")
            
            # Extract JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                scores = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            # Create scored chunks list
            scored_chunks = []
            for item in scores:
                chunk_idx = item['chunk_id'] - 1  # Convert to 0-indexed
                if 0 <= chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    chunk['relevance_score'] = float(item['score'])
                    scored_chunks.append(chunk)
            
            # Sort by score and return top-k
            scored_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            filtered = scored_chunks[:top_k]
            
            elapsed = time.time() - start_time
            debug_print("JOINT3:FILTER", f"Filtered to {len(filtered)} chunks in {elapsed:.2f}s")
            if filtered:
                avg_score = sum(c.get('relevance_score', 0) for c in filtered) / len(filtered)
                debug_print("JOINT3:FILTER", f"Average relevance score: {avg_score:.1f}/10")
            
            return filtered
            
        except Exception as e:
            debug_print("JOINT3:FILTER", f"Filtering failed: {type(e).__name__}: {e}")
            # Fallback: return original chunks (use existing scores if available)
            debug_print("JOINT3:FILTER", "Using fallback: original chunk order")
            return chunks[:top_k]

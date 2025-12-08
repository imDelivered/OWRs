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
import re
from typing import Dict, List, Tuple, Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from chatbot import config


def debug_print(joint_name: str, msg: str):
    """Print debug message for a specific joint."""
    if config.DEBUG:
        print(f"[DEBUG:{joint_name}] {msg}", file=sys.stderr)


def extract_json_from_text(text: str) -> Any:
    """
    Robustly extract the first valid JSON object or array from text.
    Handles Markdown code blocks, conversational filler, and nested structures.
    """
    if not text:
        raise ValueError("Empty text input")

    # 1. Try to find JSON in Markdown code blocks first
    code_block_pattern = r'```(?:json)?\s*(?P<content>.*?)\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        content = match.group('content')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass # Fallback to scanning raw text if block content is invalid

    # 2. Heuristic scan for looking for first '{' or '['
    # We use a simple counter to find the matching closing brace/bracket
    # This avoids issues where regex gets confused by nested braces
    
    start_indices = [m.start() for m in re.finditer(r'[\[\{]', text)]
    
    for start_idx in start_indices:
        opener = text[start_idx]
        closer = '}' if opener == '{' else ']'
        
        stack = 1
        for i in range(start_idx + 1, len(text)):
            char = text[i]
            # Simple stack logic; ignores string scraping for brevity but usually sufficient for LLM outputs
            # For a perfect parser we'd need to track string state to ignore braces inside strings, 
            # but standard json.loads will catch invalid syntax anyway.
            if char == opener:
                stack += 1
            elif char == closer:
                stack -= 1
            
            if stack == 0:
                potential_json = text[start_idx : i + 1]
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    break # Try next starting point
                    
    raise ValueError("No valid JSON found in response")


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
        Extract ALL entities from query, with comparison detection.
        
        Args:
            query: User query string
            
        Returns:
            Dict with keys: is_comparison, entities (list), action
            Each entity has: name, type, aliases
        """
        debug_print("JOINT1:ENTITY", f"Extracting entities from: '{query}'")
        start_time = time.time()
        
        prompt = f"""You are a multi-entity extraction system.

    INSTRUCTIONS:
    1. Identify ALL distinct entities (people, places, things, events) in the query.
    2. CHECK FOR COMPARISONS: If the user compares items (e.g. "vs", "compare", "difference", "and"), set "is_comparison": true.
    3. EXTRACT ALIASES:
       - Aliases must be SYNONYMS (e.g. "Biggie" -> "The Notorious B.I.G.").
       - DO NOT list related people/rivals/family as aliases.
       - Example: "Tupac" and "Biggie" are DIFFERENT. Do not list one as alias of other.

    Query: "{query}"

    Return ONLY valid JSON with this exact structure:
    {{
      "is_comparison": true,
      "entities": [
        {{"name": "Entity Name", "type": "person|place|event|concept|technology|organization", "aliases": ["valid alias"]}}
      ],
      "action": "what the user wants to know"
    }}
    """

        try:
            response = ollama_call(self.model, prompt, self.temperature, config.JOINT_TIMEOUT)
            debug_print("JOINT1:ENTITY", f"Raw response: {response[:300]}...")
            
            # Use robust extractor
            result = extract_json_from_text(response)
            
            # Handle if model returns a list wrapper
            if isinstance(result, list):
                if result:
                    result = result[0]
                else:
                    raise ValueError("Received empty list from model")
            
            # Validate new result structure
            if 'entities' not in result:
                # Try to convert old format to new format
                if 'entity' in result:
                    debug_print("JOINT1:ENTITY", "Converting old format to new multi-entity format")
                    result = {
                        'is_comparison': False,
                        'entities': [{
                            'name': result.get('entity', query),
                            'type': result.get('entity_type', 'unknown'),
                            'aliases': result.get('aliases', [])
                        }],
                        'action': result.get('action', 'information')
                    }
                else:
                    raise ValueError(f"Missing 'entities' key. Got: {result.keys()}")
            
            # Ensure entities is a list
            if not isinstance(result.get('entities'), list):
                raise ValueError(f"'entities' must be a list, got {type(result.get('entities'))}")
            
            # Ensure each entity has required keys
            for i, entity in enumerate(result['entities']):
                if 'name' not in entity:
                    raise ValueError(f"Entity {i} missing 'name' key")
                # Set defaults for optional fields
                entity.setdefault('type', 'unknown')
                entity.setdefault('aliases', [])
            
            elapsed = time.time() - start_time
            entity_names = [e['name'] for e in result['entities']]
            debug_print("JOINT1:ENTITY", f"Extracted {len(result['entities'])} entities: {entity_names}")
            debug_print("JOINT1:ENTITY", f"Is comparison: {result.get('is_comparison', False)}")
            debug_print("JOINT1:ENTITY", f"Action: {result.get('action', 'N/A')}")
            debug_print("JOINT1:ENTITY", f"Extraction took {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            debug_print("JOINT1:ENTITY", f"Extraction failed: {type(e).__name__}: {e}")
            # Fallback: return query as single entity in new format
            debug_print("JOINT1:ENTITY", "Using fallback: query as single entity")
            return {
                "is_comparison": False,
                "entities": [{
                    "name": query,
                    "type": "unknown",
                    "aliases": []
                }],
                "action": "information"
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
        Score article titles by relevance to entities.
        
        Args:
            entity_info: Entity information from EntityExtractorJoint (new multi-entity format)
            article_titles: List of Wikipedia article titles
            top_k: Return top K scored articles
            
        Returns:
            List of (title, score) tuples, sorted by score descending
        """
        if not article_titles:
            debug_print("JOINT2:SCORER", "No articles to score")
            return []
        
        # Extract all entity names and aliases from new format
        all_entity_names = []
        for entity in entity_info.get('entities', []):
            all_entity_names.append(entity.get('name', ''))
            all_entity_names.extend(entity.get('aliases', []))
        
        # Filter out empty strings
        all_entity_names = [n for n in all_entity_names if n]
        
        # Backwards compatibility: support old format
        if not all_entity_names and 'entity' in entity_info:
            all_entity_names = [entity_info['entity']] + entity_info.get('aliases', [])
        
        entities_display = [e.get('name', '') for e in entity_info.get('entities', [])]
        debug_print("JOINT2:SCORER", f"Scoring {len(article_titles)} articles for entities: {entities_display}")
        start_time = time.time()
        
        # === EXACT MATCH OVERRIDE ===
        # Before LLM scoring, check for exact entity matches
        # These get score 11.0 (above max 10) to guarantee inclusion
        exact_match_scores = []
        
        for title in article_titles:
            title_lower = title.lower().strip()
            for entity_name in all_entity_names:
                if title_lower == entity_name.lower().strip():
                    debug_print("JOINT2:SCORER", f"EXACT MATCH OVERRIDE: '{title}' == entity '{entity_name}' -> score 11.0")
                    exact_match_scores.append((title, 11.0))
                    break  # Only add once per title
        
        debug_print("JOINT2:SCORER", f"Found {len(exact_match_scores)} exact entity matches")
        
        # Format article titles for prompt (limit to prevent token overflow)
        articles_formatted = "\n".join([f"{i+1}. {title}" for i, title in enumerate(article_titles[:20])])
        
        # Format entities for prompt
        entities_str = ", ".join([f"'{e.get('name', '')}'" for e in entity_info.get('entities', [])])
        action = entity_info.get('action', 'information about')
        
        prompt = f"""I will give you a list of Article Titles.
        You must select the ones relevant to ANY of these entities: {entities_str}
        The user wants to: {action}
        
        RULES:
        1. ONLY select titles from the provided INPUT LIST below.
        2. DO NOT output example titles.
        3. Output valid JSON only.
        4. For COMPARISON queries, include articles for ALL mentioned entities.
        
        INPUT LIST:
        {articles_formatted}
        
        Rate each article 0-10 where:
        - 10 = Perfect match for one of the queried entities
        - 7-9 = Highly relevant to one of the entities
        - 0 = Not relevant to any entity
        
        Return ONLY a JSON array:
        [
          {{"title": "Actual Title From List", "score": 10}}
        ]"""

        try:
            response = ollama_call(self.model, prompt, self.temperature, config.JOINT_TIMEOUT)
            debug_print("JOINT2:SCORER", f"Raw response: {response[:200]}...")
            
            # Use robust extractor
            scores = extract_json_from_text(response)
            
            if not isinstance(scores, list):
                 raise ValueError("Response is not a JSON array")

            # --- VALIDATION & FILTERING ---
            # Helper: Normalize a title for fuzzy matching
            def normalize_title(t: str) -> str:
                """Lowercase and remove commas/punctuation for comparison."""
                return re.sub(r'[,.:;\'\"-]+', '', t.lower()).strip()
            
            def fuzzy_match(llm_title: str, candidates: List[str]) -> Optional[str]:
                """
                Try to match llm_title to a candidate using fuzzy matching.
                Returns the original candidate title if matched, None otherwise.
                """
                norm_llm = normalize_title(llm_title)
                for candidate in candidates:
                    norm_cand = normalize_title(candidate)
                    # Exact normalized match
                    if norm_llm == norm_cand:
                        return candidate
                    # Substring match (either direction)
                    if norm_cand in norm_llm or norm_llm in norm_cand:
                        return candidate
                return None
            
            # 1. Verification Set: Only allow titles that match candidates (fuzzy)
            valid_titles = list(article_titles)
            
            # 2. Placeholder Pattern: reject titles with suspicious placeholder names
            placeholder_pattern = re.compile(r'article\s+name|title\s+\d+|example\s+article', re.IGNORECASE)
            
            scored_articles = []
            for item in scores:
                llm_title = item.get('title')
                score = float(item.get('score', 0))
                
                # Check 1: Must match original list (exact or fuzzy)
                if llm_title in valid_titles:
                    matched_title = llm_title  # Exact match
                else:
                    matched_title = fuzzy_match(llm_title, valid_titles)
                    if matched_title:
                        debug_print("JOINT2:SCORER", f"Fuzzy matched: '{llm_title}' -> '{matched_title}'")
                    else:
                        debug_print("JOINT2:SCORER", f"Filtered hallucination: '{llm_title}' (not in candidates)")
                        continue
                    
                # Check 2: Must not be a placeholder
                if placeholder_pattern.search(matched_title):
                    debug_print("JOINT2:SCORER", f"Filtered placeholder: '{matched_title}'")
                    continue
                
                scored_articles.append((matched_title, score))
            
            # Sort by score
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            
            # === MERGE EXACT MATCH OVERRIDES ===
            # Prepend exact matches to ensure they're always in top results
            # Avoid duplicates by filtering out titles already in exact_match_scores
            exact_titles = {t for t, _ in exact_match_scores}
            scored_articles = [item for item in scored_articles if item[0] not in exact_titles]
            final_results = exact_match_scores + scored_articles
            
            debug_print("JOINT2:SCORER", f"After exact match merge: {len(final_results)} total articles")
            
            elapsed = time.time() - start_time
            debug_print("JOINT2:SCORER", f"Scored {len(final_results)} valid articles in {elapsed:.2f}s")
            debug_print("JOINT2:SCORER", f"Top 5 scores: {final_results[:5]}")
            
            return final_results[:top_k]
            
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
    
    def filter(self, query: str, chunks: List[Dict], top_k: int = 5, entity_info: Dict = None) -> List[Dict]:
        """
        Filter chunks by query relevance.
        
        Args:
            query: Original user query
            chunks: List of chunk dicts with 'text' and 'metadata' keys
            top_k: Return top K relevant chunks
            entity_info: Optional entity info for comparison-aware filtering
            
        Returns:
            List of chunk dicts, sorted by relevance
        """
        if not chunks:
            debug_print("JOINT3:FILTER", "No chunks to filter")
            return []
        
        # Check if this is a comparison query
        is_comparison = entity_info.get('is_comparison', False) if entity_info else False
        entities = entity_info.get('entities', []) if entity_info else []
        entity_names = [e.get('name', '').lower() for e in entities]
        
        debug_print("JOINT3:FILTER", f"Filtering {len(chunks)} chunks for query '{query}'")
        debug_print("JOINT3:FILTER", f"Is comparison: {is_comparison}, Entities: {entity_names}")
        start_time = time.time()
        
        # For comparison queries, use diversity-aware selection as primary strategy
        # This ensures we get chunks from ALL entities
        if is_comparison and len(entity_names) >= 2:
            debug_print("JOINT3:FILTER", "Using diversity-aware selection for comparison query")
            return self._diversity_filter(chunks, entity_names, top_k)
        
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
            # Use longer timeout for chunk filtering since it processes more text
            response = ollama_call(self.model, prompt, self.temperature, timeout=config.JOINT_TIMEOUT + 5)
            debug_print("JOINT3:FILTER", f"Raw response: {response[:200]}...")
            
            # Robust JSON Lines parsing helper
            def parse_json_lines(raw: str) -> List[Dict]:
                """
                Parse JSON that may be an array, a single dict, or multiple
                JSON objects separated by commas/newlines (JSON Lines).
                """
                # 1. Try standard parsing first
                try:
                    result = extract_json_from_text(raw)
                    if isinstance(result, list):
                        return result
                    if isinstance(result, dict):
                        # Got a single dict, try wrapping it
                        debug_print("JOINT3:FILTER", "Got single dict, attempting wrap")
                        pass  # Fall through to wrapping logic
                except (ValueError, json.JSONDecodeError):
                    pass  # Fall through to fallback
                
                # 2. Try wrapping the raw string in brackets
                try:
                    wrapped = f"[{raw.strip()}]"
                    result = json.loads(wrapped)
                    if isinstance(result, list):
                        debug_print("JOINT3:FILTER", "Parsed via bracket wrapping")
                        return result
                except json.JSONDecodeError:
                    pass  # Fall through to regex
                
                # 3. Regex fallback: find all JSON objects individually
                debug_print("JOINT3:FILTER", "Using regex fallback for JSON objects")
                objects = []
                # Match individual JSON objects (non-greedy)
                for match in re.finditer(r'\{[^{}]*\}', raw):
                    try:
                        obj = json.loads(match.group())
                        objects.append(obj)
                    except json.JSONDecodeError:
                        continue
                
                if objects:
                    debug_print("JOINT3:FILTER", f"Regex extracted {len(objects)} objects")
                    return objects
                
                raise ValueError("Could not parse any JSON from response")
            
            # Use robust JSON Lines parser
            scores = parse_json_lines(response)
            
            # Handle wrapper object {"chunks": [...]}
            if isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], dict) and "chunks" in scores[0]:
                scores = scores[0]["chunks"]
            elif isinstance(scores, dict) and "chunks" in scores:
                scores = scores["chunks"]
            
            if not isinstance(scores, list):
                 raise ValueError(f"Response is not a JSON array (got {type(scores).__name__})")
            
            # Create scored chunks list
            scored_chunks = []
            for item in scores:
                chunk_id = item.get('chunk_id')
                if chunk_id is None:
                    continue
                    
                chunk_idx = chunk_id - 1  # Convert to 0-indexed
                if 0 <= chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    chunk['relevance_score'] = float(item.get('score', 0))
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

    def _diversity_filter(self, chunks: List[Dict], entity_names: List[str], top_k: int) -> List[Dict]:
        """
        Diversity-aware chunk selection that ensures coverage of all entities.
        
        For comparison queries, we need chunks from EACH entity to provide
        a balanced answer.
        """
        debug_print("JOINT3:FILTER", f"Diversity filter: finding chunks for {len(entity_names)} entities")
        
        # Group chunks by which entity they likely belong to
        entity_chunks = {name: [] for name in entity_names}
        entity_chunks['other'] = []
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            chunk_title = chunk.get('metadata', {}).get('title', '').lower()
            matched = False
            
            for entity_name in entity_names:
                # Check if chunk is about this entity (by text or title)
                if entity_name in chunk_text or entity_name in chunk_title:
                    entity_chunks[entity_name].append(chunk)
                    matched = True
                    break
            
            if not matched:
                entity_chunks['other'].append(chunk)
        
        # Log distribution
        for name, c_list in entity_chunks.items():
            debug_print("JOINT3:FILTER", f"  Entity '{name}': {len(c_list)} chunks")
        
        # Allocate slots proportionally, but ensure minimum per entity
        num_entities = len(entity_names)
        min_per_entity = max(1, top_k // (num_entities + 1))  # +1 for 'other'
        slots_remaining = top_k
        
        selected = []
        
        # First pass: select min_per_entity from each entity
        for entity_name in entity_names:
            entity_list = entity_chunks[entity_name]
            to_take = min(min_per_entity, len(entity_list), slots_remaining)
            selected.extend(entity_list[:to_take])
            slots_remaining -= to_take
        
        # Second pass: fill remaining slots with best remaining chunks
        if slots_remaining > 0:
            # Collect all unused chunks
            used_ids = {id(c) for c in selected}
            remaining = []
            for name, c_list in entity_chunks.items():
                for chunk in c_list:
                    if id(chunk) not in used_ids:
                        remaining.append(chunk)
            
            # Sort by RRF score if available, otherwise just take in order
            remaining.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            selected.extend(remaining[:slots_remaining])
        
        # Add relevance scores for consistency
        for i, chunk in enumerate(selected):
            if 'relevance_score' not in chunk:
                chunk['relevance_score'] = 8.0 - (i * 0.5)  # Descending scores
        
        debug_print("JOINT3:FILTER", f"Diversity filter selected {len(selected)} chunks")
        return selected


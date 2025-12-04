"""Iterative RAG system with dynamic context pruning and verification."""

import sys
import re
from typing import Iterable, List, Optional

from kiwix_chat.models import Chunk
from kiwix_chat.rag.token_buffer import TokenBuffer
from kiwix_chat.rag.dynamic_context import DynamicContextManager
from kiwix_chat.rag.verifier import FactVerifier, VerificationResult
from kiwix_chat.rag.retriever import RAGRetriever
from kiwix_chat.rag.vector_store import get_vector_store
from kiwix_chat.chat.ollama import ollama_stream_chat


def stream_iterative_rag_response(
    model: str,
    query: str,
    messages: List[dict],
    zim_file_path: str,
    max_tokens: int = 2048,
    usage_threshold: float = 0.7,
    verification_threshold: float = 0.45,  # Lowered threshold for less restrictive filtering
    min_utilization: float = 0.8,
    max_retrieval_cycles: int = 5,
    max_response_length: int = 3000
) -> Iterable[str]:
    """Stream response using iterative RAG with dynamic context management.
    
    Main generation loop:
    1. Initialize with initial retrieval
    2. Stream tokens from LLM
    3. Accumulate text until sentence boundaries
    4. Verify sentences after completion
    5. Update context manager with generated text
    6. Trigger retrieval when chunks are pruned
    7. Yield verified text chunks
    
    Args:
        model: LLM model name
        query: User query
        messages: Chat messages
        zim_file_path: Path to ZIM file
        max_tokens: Maximum tokens in context buffer
        usage_threshold: Usage score threshold for pruning (0.0 to 1.0)
        verification_threshold: Minimum similarity for verification (0.0 to 1.0)
        min_utilization: Minimum buffer utilization before triggering retrieval
        
    Yields:
        Verified text chunks as they're generated
    """
    # Initialize components
    try:
        vector_store = get_vector_store(zim_file_path)
        retriever = RAGRetriever(zim_file_path, vector_store)
    except ImportError as e:
        print(f"[iterative-rag] ERROR: RAG dependencies not installed: {e}", file=sys.stderr)
        print(f"[iterative-rag] Install with: pip3 install --break-system-packages chromadb sentence-transformers", file=sys.stderr)
        print(f"[iterative-rag] Falling back to normal generation (no RAG)", file=sys.stderr)
        yield from ollama_stream_chat(model, messages)
        return
    except Exception as e:
        print(f"[iterative-rag] Error initializing retriever: {e}", file=sys.stderr)
        print(f"[iterative-rag] Falling back to normal generation (no RAG)", file=sys.stderr)
        yield from ollama_stream_chat(model, messages)
        return
    
    # Check if index exists
    from kiwix_chat.rag.vector_store import is_indexed
    if not is_indexed(zim_file_path):
        print(f"[iterative-rag] RAG index not found for ZIM file", file=sys.stderr)
        print(f"[iterative-rag] Build the index first: python3 kiwix_chat.py --build-index", file=sys.stderr)
        print(f"[iterative-rag] Using normal generation (no RAG)", file=sys.stderr)
        yield from ollama_stream_chat(model, messages)
        return
    
    # Initial retrieval
    try:
        initial_chunks = retriever.retrieve(query=query, top_k=5, use_hybrid=True)
        if not initial_chunks:
            print(f"[iterative-rag] No chunks found for query (index may be empty or query too specific)", file=sys.stderr)
            print(f"[iterative-rag] Using normal generation", file=sys.stderr)
            yield from ollama_stream_chat(model, messages)
            return
    except Exception as e:
        print(f"[iterative-rag] Initial retrieval failed: {e}, using normal generation", file=sys.stderr)
        yield from ollama_stream_chat(model, messages)
        return
    
    # Initialize token buffer with semantic pruning
    token_buffer = TokenBuffer(max_tokens=max_tokens, pruning_strategy="usage")
    
    # Initialize dynamic context manager
    context_manager = DynamicContextManager(
        token_buffer=token_buffer,
        retriever=retriever,
        original_query=query,
        min_utilization=min_utilization,
        usage_threshold=usage_threshold
    )
    
    # Initialize with retrieved chunks
    relevance_scores = [1.0 - (i * 0.15) for i in range(len(initial_chunks))]
    context_manager.initialize(initial_chunks, relevance_scores)
    
    # Initialize verifier
    verifier = FactVerifier()
    
    # Prepare messages with initial context
    enhanced_messages = messages.copy()
    initial_context = context_manager.get_active_context()
    
    system_msg = {
        "role": "system",
        "content": f"""Answer based on the source material provided.

Source material:
{initial_context[:2000]}"""
    }
    
    # Remove existing system messages
    enhanced_messages = [msg for msg in enhanced_messages if msg.get("role") != "system"]
    enhanced_messages.insert(0, system_msg)
    
    # Generation loop
    accumulated_text = ""
    sentence_buffer = ""
    pending_output = ""  # Text that's been verified and ready to yield
    verified_output = ""  # All verified output so far (for internal contradiction checking)
    MAX_VERIFIED_OUTPUT_SIZE = 50000  # Limit verified_output to prevent unbounded growth
    retrieval_cycle_count = 0
    last_context_update_length = 0
    contradiction_count = 0  # Track number of contradictions (tolerance system)
    MAX_CONTRADICTION_TOLERANCE = 2  # Allow up to 2 contradictions before blocking
    
    for token_chunk in ollama_stream_chat(model, enhanced_messages):
        # Stop if response is too long
        if len(accumulated_text) > max_response_length:
            print(f"[iterative-rag] Response length limit reached ({max_response_length} chars), stopping generation", file=sys.stderr)
            break
        accumulated_text += token_chunk
        sentence_buffer += token_chunk
        
        # Check for sentence boundaries
        sentence_end_pattern = re.compile(r'[.!?]+\s+')
        sentences = sentence_end_pattern.split(sentence_buffer)
        
        # If we have complete sentences, verify them
        if len(sentences) > 1:
            # Process all complete sentences (except the last incomplete one)
            complete_sentences = sentences[:-1]
            incomplete_sentence = sentences[-1] if sentences else ""
            
            for sentence in complete_sentences:
                if not sentence.strip():
                    continue
                
                # Get current source chunks
                source_chunks = context_manager.get_source_chunks()
                
                # Verify sentence
                verification = verifier.verify_sentence(
                    sentence=sentence.strip(),
                    source_chunks=source_chunks,
                    similarity_threshold=verification_threshold
                )
                
                # Check for contradictions (both against source and internal)
                has_major_contradiction = False
                has_internal_contradiction = False
                
                if source_chunks:
                    sentence_lower = sentence.lower()
                    source_text = " ".join([chunk.text.lower() for chunk in source_chunks])
                    verified_lower = verified_output.lower()
                    
                    # Generic contradiction check: if response directly contradicts source
                    # Only check for obvious factual contradictions, not domain-specific knowledge
                    
                    # Check for internal contradictions (dates, names, locations)
                    # More lenient: only block if dates refer to the same event/context
                    if verified_lower:
                        # Extract dates from current sentence and verified output
                        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
                        sentence_dates = re.findall(date_pattern, sentence, re.IGNORECASE)
                        verified_dates = re.findall(date_pattern, verified_output, re.IGNORECASE)
                        
                        # Check if dates contradict - only if they appear to refer to the same event
                        if sentence_dates and verified_dates:
                            sentence_date_norm = re.sub(r'[,\s]+', ' ', sentence_dates[0]).strip().lower()
                            verified_date_norm = re.sub(r'[,\s]+', ' ', verified_dates[0]).strip().lower()
                            if sentence_date_norm != verified_date_norm:
                                # Check if dates refer to the same event by looking for shared context
                                # Extract key terms around dates (10 words before/after)
                                sentence_words = sentence.lower().split()
                                verified_words = verified_output.lower().split()
                                
                                # Find date positions and extract context
                                sentence_date_idx = sentence.lower().find(sentence_dates[0].lower())
                                verified_date_idx = verified_output.lower().find(verified_dates[0].lower())
                                
                                # Extract surrounding context (20 chars before/after)
                                sentence_context = sentence[max(0, sentence_date_idx-20):sentence_date_idx+len(sentence_dates[0])+20].lower()
                                verified_context = verified_output[max(0, verified_date_idx-20):verified_date_idx+len(verified_dates[0])+20].lower()
                                
                                # Check for shared event keywords (death, birth, shot, fight, etc.)
                                event_keywords = ['died', 'death', 'shot', 'killed', 'born', 'birth', 'fight', 'match', 'event', 'occurred', 'happened']
                                sentence_has_event = any(kw in sentence_context for kw in event_keywords)
                                verified_has_event = any(kw in verified_context for kw in event_keywords)
                                
                                # Only block if both dates have event keywords AND share similar context
                                # This prevents blocking different events that happen to mention dates
                                if sentence_has_event and verified_has_event:
                                    # Check for shared entity names (person, place) in context
                                    # Simple heuristic: if both contexts mention the same proper noun, likely same event
                                    proper_nouns_sentence = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence_context)
                                    proper_nouns_verified = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', verified_context)
                                    
                                    shared_entities = set(p.lower() for p in proper_nouns_sentence) & set(p.lower() for p in proper_nouns_verified)
                                    # Filter out common words
                                    common_words = {'the', 'this', 'that', 'these', 'those', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
                                    shared_entities = {e for e in shared_entities if e not in common_words and len(e) > 3}
                                    
                                    # Only block if there's clear shared context (same entity/event)
                                    if shared_entities:
                                        contradiction_count += 1
                                        if contradiction_count > MAX_CONTRADICTION_TOLERANCE:
                                            has_internal_contradiction = True
                                            print(f"[iterative-rag] ❌ INTERNAL CONTRADICTION: Date '{sentence_dates[0]}' contradicts earlier date '{verified_dates[0]}' (contradiction #{contradiction_count}, threshold exceeded)", file=sys.stderr)
                                        else:
                                            print(f"[iterative-rag] ⚠ Date difference noted: '{sentence_dates[0]}' vs '{verified_dates[0]}' (tolerance: {contradiction_count}/{MAX_CONTRADICTION_TOLERANCE})", file=sys.stderr)
                                    else:
                                        # Different events, allow both dates
                                        print(f"[iterative-rag] ℹ Different events with different dates - allowing both", file=sys.stderr)
                                else:
                                    # No clear event context, likely different events - allow
                                    print(f"[iterative-rag] ℹ Date difference without clear event context - allowing", file=sys.stderr)
                        
                    # Check for duplicate sentences (prevent same content appearing twice)
                    sentence_normalized = re.sub(r'[^\w\s]', '', sentence.strip().lower())
                    # Check if this sentence (or very similar) already appeared
                    verified_sentences = re.split(r'[.!?]+\s+', verified_output)
                    for prev_sentence in verified_sentences:
                        if len(prev_sentence.strip()) > 20:  # Only check substantial sentences
                            prev_normalized = re.sub(r'[^\w\s]', '', prev_sentence.strip().lower())
                            # Check similarity (simple: if >80% words match, consider duplicate)
                            sentence_words = set(sentence_normalized.split())
                            prev_words = set(prev_normalized.split())
                            if len(sentence_words) > 5 and len(prev_words) > 5:  # Only for substantial sentences
                                overlap = len(sentence_words & prev_words)
                                similarity = overlap / max(len(sentence_words), len(prev_words))
                                if similarity > 0.80:  # 80% word overlap = likely duplicate
                                    print(f"[iterative-rag] ⚠ DUPLICATE DETECTED: Similar sentence already output (similarity: {similarity:.2f})", file=sys.stderr)
                                    has_internal_contradiction = True  # Block duplicates
                                    break
                
                # Add to pending output only if verified and no contradictions
                if verification.is_verified and not has_major_contradiction and not has_internal_contradiction:
                    pending_output += sentence.strip() + ". "
                    verified_output += sentence.strip() + ". "  # Track verified output
                    # Limit verified_output size to prevent unbounded growth
                    if len(verified_output) > MAX_VERIFIED_OUTPUT_SIZE:
                        # Keep last portion for contradiction checking
                        verified_output = verified_output[-MAX_VERIFIED_OUTPUT_SIZE:]
                elif has_major_contradiction or has_internal_contradiction:
                    # Block contradictory sentences
                    print(f"[iterative-rag] ⛔ BLOCKED contradictory sentence: {sentence[:100]}...", file=sys.stderr)
                    # Don't add to output - skip this sentence
                elif verification.confidence < 0.30:
                    # Low confidence - warn and skip (lowered threshold from 0.5 to 0.30)
                    print(f"[iterative-rag] ⚠ Low confidence verification ({verification.confidence:.2f}), BLOCKING: {sentence[:80]}...", file=sys.stderr)
                    if verification.claims:
                        print(f"[iterative-rag]   Claims checked: {', '.join(verification.claims[:3])}", file=sys.stderr)
                    # Don't add to output
                else:
                    # Medium confidence - allow but warn
                    print(f"[iterative-rag] ⚠ Medium confidence ({verification.confidence:.2f}): {sentence[:80]}...", file=sys.stderr)
                    pending_output += sentence.strip() + ". "
            
            # Update context manager with generated text
            # This triggers usage score updates and pruning
            context_state = context_manager.add_generated_text(accumulated_text)
            
            # Update system message with new context if it changed significantly
            # But limit retrieval cycles and only update if context actually changed
            if context_state.needs_retrieval and retrieval_cycle_count < max_retrieval_cycles:
                new_context = context_manager.get_active_context()
                # Only update if context changed significantly (more than 20% different)
                if len(new_context) != last_context_update_length:
                    retrieval_cycle_count += 1
                    last_context_update_length = len(new_context)
                    # Update system message
                    enhanced_messages[0]["content"] = f"""Answer based on the source material provided.

Source material:
{new_context[:2000]}"""
                elif retrieval_cycle_count >= max_retrieval_cycles:
                    print(f"[iterative-rag] Maximum retrieval cycles reached ({max_retrieval_cycles}), continuing with current context", file=sys.stderr)
            
            # Reset sentence buffer to incomplete sentence
            sentence_buffer = incomplete_sentence
        
        # Yield pending output periodically (every N tokens or when buffer gets large)
        if len(pending_output) > 100:
            yield pending_output
            # Note: verified_output already includes this (updated when adding to pending_output)
            pending_output = ""
    
    # Process remaining sentence buffer
    if sentence_buffer.strip():
        source_chunks = context_manager.get_source_chunks()
        
        # Check for contradictions
        sentence_lower = sentence_buffer.lower()
        source_text = " ".join([chunk.text.lower() for chunk in source_chunks]) if source_chunks else ""
        verified_lower = verified_output.lower()
        
        has_contradiction = False
        # Generic contradiction check only - no domain-specific knowledge
        
        # Check internal date contradiction (using same lenient logic)
        if verified_lower:
            date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            sentence_dates = re.findall(date_pattern, sentence_buffer, re.IGNORECASE)
            verified_dates = re.findall(date_pattern, verified_output, re.IGNORECASE)
            if sentence_dates and verified_dates:
                sentence_date_norm = re.sub(r'[,\s]+', ' ', sentence_dates[0]).strip().lower()
                verified_date_norm = re.sub(r'[,\s]+', ' ', verified_dates[0]).strip().lower()
                if sentence_date_norm != verified_date_norm:
                    # Apply same lenient logic as above
                    sentence_date_idx = sentence_buffer.lower().find(sentence_dates[0].lower())
                    verified_date_idx = verified_output.lower().find(verified_dates[0].lower())
                    sentence_context = sentence_buffer[max(0, sentence_date_idx-20):sentence_date_idx+len(sentence_dates[0])+20].lower()
                    verified_context = verified_output[max(0, verified_date_idx-20):verified_date_idx+len(verified_dates[0])+20].lower()
                    event_keywords = ['died', 'death', 'shot', 'killed', 'born', 'birth', 'fight', 'match', 'event', 'occurred', 'happened']
                    sentence_has_event = any(kw in sentence_context for kw in event_keywords)
                    verified_has_event = any(kw in verified_context for kw in event_keywords)
                    if sentence_has_event and verified_has_event:
                        proper_nouns_sentence = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence_context)
                        proper_nouns_verified = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', verified_context)
                        shared_entities = set(p.lower() for p in proper_nouns_sentence) & set(p.lower() for p in proper_nouns_verified)
                        common_words = {'the', 'this', 'that', 'these', 'those', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
                        shared_entities = {e for e in shared_entities if e not in common_words and len(e) > 3}
                        if shared_entities:
                            contradiction_count += 1
                            if contradiction_count > MAX_CONTRADICTION_TOLERANCE:
                                has_contradiction = True
        
        verification = verifier.verify_sentence(
            sentence=sentence_buffer.strip(),
            source_chunks=source_chunks,
            similarity_threshold=verification_threshold
        )
        
        if verification.is_verified and not has_contradiction:
            pending_output += sentence_buffer.strip()
            verified_output += sentence_buffer.strip()
            # Limit verified_output size
            if len(verified_output) > MAX_VERIFIED_OUTPUT_SIZE:
                verified_output = verified_output[-MAX_VERIFIED_OUTPUT_SIZE:]
        elif has_contradiction:
            print(f"[iterative-rag] ⛔ BLOCKED final sentence due to contradiction", file=sys.stderr)
        
        # Final context update
        context_manager.add_generated_text(accumulated_text)
    
    # Yield any remaining output
    if pending_output:
        yield pending_output


def iterative_rag_query(
    model: str,
    query: str,
    zim_file_path: str,
    conversation_history: Optional[List[dict]] = None,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048
) -> Iterable[str]:
    """Simple interface for iterative RAG query.
    
    Args:
        model: LLM model name
        query: User query
        zim_file_path: Path to ZIM file
        conversation_history: Optional conversation history
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in context buffer
        
    Yields:
        Response text chunks (streaming in real-time)
    """
    messages = conversation_history.copy() if conversation_history else []
    
    # Add system prompt if provided
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages[0]["content"] = system_prompt
    
    messages.append({"role": "user", "content": query})
    
    yield from stream_iterative_rag_response(
        model=model,
        query=query,
        messages=messages,
        zim_file_path=zim_file_path,
        max_tokens=max_tokens
    )


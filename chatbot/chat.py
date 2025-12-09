"""Ollama chat API integration."""

import sys
import json
from typing import List, Iterable
from urllib.request import Request, urlopen
from urllib.error import URLError

from chatbot.config import OLLAMA_CHAT_URL
from chatbot.models import Message
from chatbot import config



def debug_print(msg: str):
    if config.DEBUG:
        print(f"[DEBUG:CHAT] {msg}", file=sys.stderr)


# Global status callback for UI updates
_status_callback = None

def set_status_callback(callback):
    """Set a callback function to receive status updates during RAG processing."""
    global _status_callback
    _status_callback = callback

def _update_status(status: str):
    """Call the status callback if set."""
    global _status_callback
    if _status_callback:
        try:
            _status_callback(status)
        except:
            pass


def stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """Stream chat with Ollama model."""
    debug_print(f"stream_chat called with model='{model}'")
    debug_print(f"stream_chat message count: {len(messages)}")
    payload = json.dumps({
        "model": model, 
        "messages": messages, 
        "stream": True,
        "options": {"temperature": 0}
    }).encode("utf-8")
    debug_print(f"Payload size: {len(payload)} bytes")
    debug_print(f"Sending request to: {OLLAMA_CHAT_URL}")
    req = Request(OLLAMA_CHAT_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        debug_print("Opening connection to Ollama...")
        with urlopen(req, timeout=60) as resp:
            debug_print(f"Connection established, status: {resp.status}")
            token_count = 0
            for raw_line in resp:
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    debug_print("Warning: Failed to decode JSON line")
                    continue
                if obj.get("error"):
                    debug_print(f"Error from Ollama: {obj['error']}")
                    raise RuntimeError(str(obj["error"]))
                message = obj.get("message", {})
                content_piece = message.get("content", "")
                if content_piece:
                    token_count += 1
                    yield content_piece
                if obj.get("done"):
                    debug_print(f"Stream complete. Total tokens yielded: {token_count}")
                    break
    except URLError as e:
        debug_print(f"URLError: {e.reason}")
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


def full_chat(model: str, messages: List[dict]) -> str:
    """Full chat with Ollama model."""
    debug_print(f"full_chat called with model='{model}'")
    debug_print(f"full_chat message count: {len(messages)}")
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0}
    }).encode("utf-8")
    debug_print(f"Payload size: {len(payload)} bytes")
    debug_print(f"Sending request to: {OLLAMA_CHAT_URL}")
    req = Request(OLLAMA_CHAT_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        debug_print("Opening connection to Ollama...")
        with urlopen(req, timeout=60) as resp:
            debug_print(f"Connection established, status: {resp.status}")
            data = resp.read()
            debug_print(f"Response size: {len(data)} bytes")
            obj = json.loads(data.decode("utf-8"))
            if obj.get("error"):
                debug_print(f"Error from Ollama: {obj['error']}")
                raise RuntimeError(str(obj["error"]))
            message = obj.get("message", {})
            response_content = message.get("content", "")
            debug_print(f"Response length: {len(response_content)} characters")
            return response_content
    except URLError as e:
        debug_print(f"URLError: {e.reason}")
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


from chatbot.rag import RAGSystem

# Global RAG instance
_rag_system = None

def get_rag_system():
    global _rag_system
    debug_print("get_rag_system called")
    if _rag_system is None:
        debug_print("RAG system not initialized, checking for resources...")
        # Initialize only if index exists
        import os
        if os.path.exists("data/index/faiss.index") or os.path.exists("wikipedia_en_all_maxi_2025-08.zim") or any(f.endswith(".zim") for f in os.listdir('.')):
            try:
                print("Initializing RAG system (Hybrid/Fast)...")
                _rag_system = RAGSystem()
                _rag_system.load_resources()
                debug_print("RAG system initialized successfully")
            except Exception as e:
                print(f"Failed to load RAG: {e}")
                debug_print(f"RAG initialization failed: {e}")
                _rag_system = None
        else:
            debug_print("No RAG resources found (no index or ZIM files)")
    else:
        debug_print("RAG system already initialized")
    return _rag_system

def build_messages(system_prompt: str, history: List[Message], user_query: str = None) -> List[dict]:
    """Build message list for Ollama API with RAG augmentation."""
    debug_print("="*60)
    debug_print("build_messages START")
    debug_print(f"system_prompt length: {len(system_prompt)} chars")
    debug_print(f"history length: {len(history)} messages")
    debug_print(f"user_query: '{user_query}'")
    
    # 1. Retrieve context if we have a user query
    context_text = ""
    rag = get_rag_system()
    
    # 1. Detect Intent
    from chatbot.intent import detect_intent
    # Identify the actual query. If user_query is provided, use it.
    # Otherwise check the last message in history if it's from user.
    query_text = user_query
    if not query_text and history and history[-1].role == 'user':
        query_text = history[-1].content
        debug_print(f"Using last message from history as query: '{query_text}'")
    else:
        debug_print(f"Using provided user_query: '{query_text}'")
        
    intent = detect_intent(query_text or "")
    debug_print(f"Intent Detection Result: mode='{intent.mode_name}', should_retrieve={intent.should_retrieve}")
    _update_status("Analyzing query")
    
    # 2. Retrieve context (If Intent allows)
    debug_print("-" * 60)
    debug_print("RAG RETRIEVAL PHASE")
    context_text = ""
    rag = get_rag_system()
        
    if rag and query_text and intent.should_retrieve:
        debug_print(f"Conditions met for RAG retrieval: rag={rag is not None}, query_text='{query_text}', should_retrieve={intent.should_retrieve}")
        try:
            _update_status("Searching knowledge base")
            debug_print(f"Calling rag.retrieve with query='{query_text}', top_k=8")
            results = rag.retrieve(query_text, top_k=8)
            _update_status("Processing results")
            debug_print(f"RAG retrieve returned {len(results)} results")
            
            if results:
                debug_print("Processing RAG results...")
                context_text = "\n\nRelevant Context via RAG:\n"
                for i, r in enumerate(results, 1):
                    meta = r['metadata']
                    text = r['text']
                    title = meta.get('title', 'Unknown')
                    score = r.get('score', 0.0)
                    debug_print(f"Result {i}: title='{title}', score={score:.4f}, text_length={len(text)} chars")
                    context_text += f"\n--- Source {i}: {title} ---\n{text}\n"
                
                context_text += f"\n\nCRITICAL INSTRUCTIONS FOR PROCESSING CONTEXT:\n" \
                                f"STEP 1 - IDENTIFY THE QUESTION TYPE: Determine what specific fact or information is requested.\n" \
                                f"  - For factual questions (who/what/when/where): Look for direct statements or clear implications.\n" \
                                f"  - For descriptive questions: Synthesize information from multiple sources when available.\n" \
                                f"\n" \
                                f"STEP 2 - SEARCH SOURCES: Carefully scan all provided context for the answer.\n" \
                                f"  - Look for direct statements (e.g., 'Paris is the capital of France').\n" \
                                f"  - Also recognize equivalent phrasings (e.g., 'the capital is Paris' or 'France's capital, Paris').\n" \
                                f"  - Pay attention to context to avoid confusion (e.g., 'capital' meaning city vs. financial capital).\n" \
                                f"\n" \
                                f"STEP 3 - EXTRACT or REFUSE:\n" \
                                f"  - If the answer is directly stated OR clearly implied by the context: Extract and provide it.\n" \
                                f"  - You may infer when the inference is obvious (e.g., if text says 'Paris is the capital of France', this answers 'What is the capital of France?').\n" \
                                f"  - If the context discusses the topic but does NOT contain information that answers the specific question: Refuse.\n" \
                                f"  - DO NOT use external knowledge beyond what's in the provided context.\n" \
                                f"  - DO NOT guess or make up facts.\n" \
                                f"\n" \
                                f"STEP 4 - RESPOND:\n" \
                                f"  - If the answer is found: Provide a clear, direct answer and cite the source(s).\n" \
                                f"  - If NOT found: Explicitly state 'I do not have enough information in the provided context to answer this question.'\n" \
                                f"\n" \
                                f"REMEMBER: Balance accuracy with helpfulness. Answer when context clearly supports it, but refuse when it doesn't."
                debug_print(f"Context assembled: {len(context_text)} chars total")
            else:
                debug_print("No results returned from RAG")
                if config.STRICT_RAG_MODE:
                    debug_print("STRICT_RAG_MODE=True, will refuse to answer")
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index.\n" \
                                   "Instructions: You MUST refuse to answer the user's question because no relevant context was found.\n" \
                                   "Reply EXACTLY with: 'I do not have enough information in my knowledge base to answer this question.'"
                else:
                    debug_print("STRICT_RAG_MODE=False, will use general knowledge")
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index. Answering based on general knowledge.\n"
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            debug_print(f"RAG retrieval exception: {type(e).__name__}: {e}")
    else:
        debug_print(f"Skipping RAG retrieval: rag={rag is not None}, query_text={bool(query_text)}, should_retrieve={intent.should_retrieve}")

    # 3. Augment system prompt with Context AND Intent Instructions
    debug_print("-" * 60)
    debug_print("MESSAGE CONSTRUCTION PHASE")
    final_system_prompt = system_prompt + intent.system_instruction
    debug_print(f"Base system_prompt + intent instruction = {len(final_system_prompt)} chars")
    if context_text:
        final_system_prompt += context_text
        debug_print(f"Added context. Final system_prompt = {len(final_system_prompt)} chars")
    else:
        debug_print("No context to add")

    messages = [{"role": "system", "content": final_system_prompt}]
    debug_print(f"Added system message (length: {len(final_system_prompt)} chars)")
    
    for msg in history:
        if msg.role in ["user", "assistant", "system"]:
            messages.append({"role": msg.role, "content": msg.content})
            debug_print(f"Added {msg.role} message (length: {len(msg.content)} chars)")
            
    debug_print(f"Total messages constructed: {len(messages)}")
    debug_print("build_messages END")
    debug_print("="*60)
    print(f"\nGenerating response...")
    return messages

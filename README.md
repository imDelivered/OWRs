# KiwixRAG

A powerful offline-capable chatbot with **Retrieval-Augmented Generation (RAG)** that lets you chat with AI using local knowledge bases like Wikipedia, Python documentation, or any ZIM file archive.

> **⚠️ Platform Note:** This software is currently only available for Linux. Windows and macOS support may be added in the future.

---

## Quick Setup

### Prerequisites

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Linux (tested on Ubuntu/Debian)                                           │
│ • Python 3.8+                                                               │
│ • Internet connection (for initial setup)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Installation Steps

**Step 1: Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

> **What the setup script does:**
> - Installs Python and dependencies
> - Sets up a virtual environment
> - Installs Ollama
> - Downloads AI models:
>   - llama3.2:1b (default response model + Joint 1 & 3)
>   - llama3.1:1b (alternative response model, optional)
>   - qwen2.5:0.5b (Joint 2: article scoring)
> - Enables Ollama service
> - Creates a `krag` command for easy access

**Step 2: Add a ZIM file (optional but recommended)**
- Download a ZIM file (e.g., from [Kiwix](https://library.kiwix.org/))
- Place it in the project directory (e.g., `wikipedia_en_all_maxi_2025-08.zim`)
- The chatbot will automatically detect and use it

**Step 3: Start the chatbot**
```bash
krag
```

Or run manually:
```bash
./run_chatbot.sh
```

<img width="895" height="700" alt="Screenshot_20251205_173826" src="https://github.com/user-attachments/assets/f4509a75-dd30-4344-af13-f1a5d5c293b6" />

---

## Disclaimer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DISCLAIMER OF LIABILITY                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ This software is provided "as is" without warranty of any kind. The        │
│ author(s) and contributors are not responsible for any misuse, damage, or   │
│ consequences arising from the use of this software.                         │
│                                                                             │
│ Users are solely responsible for:                                           │
│ • Compliance with applicable laws and regulations                           │
│ • Ethical use of AI technology                                              │
│ • Content generated or accessed through this software                        │
│ • Any actions taken based on information from this software                 │
│ • Verifying the accuracy of AI-generated content                            │
│                                                                             │
│ By using this software, you agree to use it responsibly and accept full     │
│ liability for your actions.                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • AI Chat Interface: Beautiful GUI built with tkinter                      │
│ • Offline Knowledge Base: Works with ZIM files                             │
│ • Hybrid Search: Semantic (FAISS) + keyword (BM25) search                  │
│ • Just-In-Time Indexing: Auto-indexes articles on-the-fly                  │
│ • Modern UI: Dark/light mode, autocomplete, shortcuts                      │
│ • Multiple Models: Switch between any Ollama model                          │
│ • Multi-Joint RAG: Three reasoning models prevent hallucinations           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

KiwixRAG uses a **multi-joint architecture** where three small AI reasoning models work together to prevent hallucinations and ensure accurate responses.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-JOINT RAG PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Query → JOINT 1 (Entity Extraction) → Dual-Path Search              │
│       ↓                                                                     │
│  JOINT 2 (Article Scoring) → Just-In-Time Indexing → Hybrid Retrieval      │
│       ↓                    (Hugging Face: all-MiniLM-L6-v2)                 │
│  JOINT 3 (Chunk Filtering) → Final LLM Generation                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

![Architecture Diagram](architecture_diagram.png)

### Reasoning Joints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  JOINT 1: Entity Extraction (llama3.2:1b)                                 │
│  JOINT 2: Article Scoring (qwen2.5:0.5b)                                   │
│  JOINT 3: Chunk Filtering (llama3.2:1b)                                    │
│                                                                             │
│  These three models work together to extract entities, score articles,     │
│  and filter chunks, ensuring only relevant information reaches the LLM.    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  • Embeddings: Hugging Face all-MiniLM-L6-v2 (semantic search)              │
│  • Hybrid Retrieval: FAISS (vector) + BM25 (keyword) search                │
│  • Just-In-Time Indexing: Articles indexed on-the-fly as needed            │
│  • Modern GUI: Streaming responses, autocomplete, dark/light mode          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Manual Setup (Alternative)

> **If you prefer manual setup instead of the automated script:**

```bash
# Install system dependencies
sudo apt install python3 python3-venv python3-tk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:1b

# Run the chatbot
python3 run_chatbot.py
```

---

## Usage

### Basic Commands

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Command          │ Action                                                    │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ /help            │ Show help menu                                            │
│ /clear           │ Clear chat history                                        │
│ /dark            │ Toggle dark/light mode                                    │
│ /model           │ Switch to a different Ollama model                        │
│ /exit or :q      │ Quit the application                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Shortcut              │ Action                                               │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ Enter                 │ Send message                                         │
│ Highlight + Enter     │ Auto-paste and query selected text                  │
│ Ctrl+Click            │ Select word and query it                             │
│ ↑↓                    │ Navigate autocomplete suggestions                    │
│ Tab                   │ Select autocomplete suggestion                       │
│ Esc                   │ Close dialogs                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Building a Full Index (Optional)

For faster retrieval on large ZIM files, you can pre-build an index:

```bash
python3 build_index.py --zim wikipedia_en_all_maxi_2025-08.zim
```

> **Note:** This creates a full FAISS + BM25 index in `data/index/`. Without this, the system uses Just-In-Time indexing (works great, just slightly slower on first queries).

---

## Configuration

### Default Model

Edit `chatbot/config.py` to change the default model:

```python
DEFAULT_MODEL = "llama3.2:1b"  # Change to your preferred model
```

### RAG Settings

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Setting              │ Value                                                 │
├──────────────────────┼───────────────────────────────────────────────────────┤
│ Embedding Model      │ all-MiniLM-L6-v2 (fast, efficient)                    │
│ Top-K Results        │ 5 chunks per query (configurable)                    │
│ Chunk Size           │ 500 words with 50-word overlap                       │
│ Joint System         │ Enabled by default (can disable in config.py)         │
│ Strict RAG Mode      │ Enabled (requires context to answer)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

> Configurable in `chatbot/rag.py` and `chatbot/config.py`

---

## Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Package                │ Purpose                                            │
├────────────────────────┼────────────────────────────────────────────────────┤
│ libzim                 │ ZIM file reading                                   │
│ sentence-transformers  │ Text embeddings                                    │
│ faiss-cpu              │ Vector similarity search                           │
│ rank_bm25              │ Keyword search                                     │
│ beautifulsoup4         │ HTML parsing                                       │
│ numpy                  │ Numerical operations                               │
│ tqdm                   │ Progress bars                                      │
│ requests               │ HTTP requests                                     │
│ ollama                 │ Ollama API client                                  │
│ tkinter                │ GUI (usually pre-installed with Python)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> All dependencies are installed automatically by `setup.sh`.

---

## Use Cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Offline Wikipedia    │ Ask questions without internet                     │
│ • Documentation Chat   │ Chat with Python docs, manuals                     │
│ • Research Assistant   │ Query large knowledge bases locally                │
│ • Educational Tool     │ Learn from offline encyclopedias                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Note:** While this system can help explore knowledge bases, remember that AI responses may contain errors or fabricated information. Always cross-reference important facts with authoritative sources.

---

## Troubleshooting

### "tkinter not available"
```bash
sudo apt install python3-tk
```

### "Cannot reach Ollama"
Make sure Ollama is running:
```bash
ollama serve
```

### "No models found"
Install required models:
```bash
ollama pull llama3.2:1b  # Default model + joints
ollama pull qwen2.5:0.5b  # Article scoring joint
```

### Slow first queries
> This is normal with Just-In-Time indexing. The system indexes articles as needed. For faster performance, build a full index (see above).

---

## Notes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ • Works without ZIM file, but no RAG capabilities                           │
│ • First-time setup downloads ~1-2GB (Ollama + models + deps)                │
│ • GPU acceleration automatic if CUDA available                              │
│ • All data stays local - no internet required after setup                  │
│ • Joint system adds ~1.3s latency but significantly improves accuracy      │
│ • Can disable joints in config.py for faster (but less accurate) responses │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Resources

- [Ollama Models](https://ollama.ai/library)
- [Kiwix ZIM Files](https://library.kiwix.org/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

## License

See repository for license information.

---

**Enjoy chatting with your offline AI assistant!**

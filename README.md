# KiwixRAG

A powerful offline-capable chatbot with **Retrieval-Augmented Generation (RAG)** that lets you chat with AI using local knowledge bases like Wikipedia, Python documentation, or any ZIM file archive.

---

## Disclaimer

```
┌──────────────────────────────────────────────────────────────┐
│ DISCLAIMER OF LIABILITY                                       │
├──────────────────────────────────────────────────────────────┤
│ This software is provided "as is" without warranty of any    │
│ kind. The author(s) and contributors are not responsible for │
│ any misuse, damage, or consequences arising from the use of  │
│ this software.                                                │
│                                                              │
│ Users are solely responsible for:                            │
│ • Compliance with applicable laws and regulations            │
│ • Ethical use of AI technology                                │
│ • Content generated or accessed through this software         │
│ • Any actions taken based on information from this software  │
│                                                              │
│ By using this software, you agree to use it responsibly and  │
│ accept full liability for your actions.                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Features

```
┌─────────────────────────────────────────────────────────────────┐
│ • AI Chat Interface: Beautiful GUI built with tkinter          │
│ • Offline Knowledge Base: Works with ZIM files                  │
│ • Hybrid Search: Semantic (FAISS) + keyword (BM25) search      │
│ • Just-In-Time Indexing: Auto-indexes articles on-the-fly      │
│ • Modern UI: Dark/light mode, autocomplete, shortcuts          │
│ • Multiple Models: Switch between any Ollama model              │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Architecture Overview

> **RAG System Flow:**
> 1. User asks a question
> 2. System searches ZIM archive using hybrid search
> 3. Retrieves most relevant text chunks
> 4. Augments AI context with retrieved information
> 5. Generates accurate, source-cited responses

### Components

**1. Hybrid Retrieval**
```
┌─────────────────────────────────────────────────────────────┐
│ Dense Search (FAISS)    → Semantic similarity via embeddings │
│ Sparse Search (BM25)    → Keyword-based matching             │
│ Reciprocal Rank Fusion  → Combines both for optimal results  │
└─────────────────────────────────────────────────────────────┘
```

**2. Just-In-Time Indexing**
- No need to pre-index the entire ZIM file
- Articles are indexed automatically when relevant to your query 
- Fast startup, efficient memory usage

**3. GUI Features**
- Real-time streaming responses
- Query history and autocomplete
- Text selection and quick queries (Ctrl+Click, highlight+Enter)
- Model switching without restarting

---

## Quick Setup

### Prerequisites

```
┌─────────────────────────────────────────┐
│ • Linux (tested on Ubuntu/Debian)       │
│ • Python 3.8+                           │
│ • Internet connection (for initial setup)│
└─────────────────────────────────────────┘
```

### Installation Steps

**Step 1: Clone or download this repository**
```bash
cd /path/to/kiwixrag
```

**Step 2: Run the setup script**
```bash
chmod +x setup.sh
./setup.sh
```

> **What the setup script does:**
> - Installs Python and dependencies
> - Sets up a virtual environment
> - Installs Ollama
> - Downloads the default AI model (llama3.2:1b)
> - Enables Ollama service
> - Creates a `krag` command for easy access

**Step 3: Add a ZIM file (optional but recommended)**
- Download a ZIM file (e.g., from [Kiwix](https://library.kiwix.org/))
- Place it in the project directory (e.g., `wikipedia_en_all_maxi_2025-08.zim`)
- The chatbot will automatically detect and use it

**Step 4: Start the chatbot**
```bash
krag
```

Or manually:
```bash
./run_chatbot.sh
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
┌──────────────────────────────────────────────────────────────┐
│ Command          │ Action                                     │
├──────────────────┼───────────────────────────────────────────┤
│ /help            │ Show help menu                             │
│ /clear           │ Clear chat history                         │
│ /dark            │ Toggle dark/light mode                     │
│ /model           │ Switch to a different Ollama model         │
│ /exit or :q      │ Quit the application                       │
└──────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

```
┌──────────────────────────────────────────────────────────────┐
│ Enter              │ Send message                              │
│ Highlight + Enter  │ Auto-paste and query selected text       │
│ Ctrl+Click         │ Select word and query it                  │
│ ↑↓                 │ Navigate autocomplete suggestions          │
│ Tab                │ Select autocomplete suggestion            │
│ Esc                │ Close dialogs                             │
└──────────────────────────────────────────────────────────────┘
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
┌──────────────────────────────────────────────────────────────┐
│ Setting           │ Value                                    │
├──────────────────┼───────────────────────────────────────────┤
│ Embedding Model   │ all-MiniLM-L6-v2 (fast, efficient)       │
│ Top-K Results     │ 3 chunks per query                        │
│ Chunk Size        │ 500 words with 50-word overlap           │
└──────────────────────────────────────────────────────────────┘
```

> Configurable in `chatbot/rag.py`

---

## Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│ libzim              │ ZIM file reading                        │
│ sentence-transformers│ Text embeddings                         │
│ faiss-cpu           │ Vector similarity search                 │
│ rank_bm25           │ Keyword search                           │
│ beautifulsoup4      │ HTML parsing                             │
│ tkinter             │ GUI (usually pre-installed with Python)  │
└──────────────────────────────────────────────────────────────┘
```

> All dependencies are installed automatically by `setup.sh`.

---

## Use Cases

```
┌──────────────────────────────────────────────────────────────┐
│ • Offline Wikipedia    │ Ask questions without internet       │
│ • Documentation Chat   │ Chat with Python docs, manuals       │
│ • Research Assistant   │ Query large knowledge bases locally   │
│ • Educational Tool     │ Learn from offline encyclopedias      │
└──────────────────────────────────────────────────────────────┘
```

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
Install a model:
```bash
ollama pull llama3.2:1b
```

### Slow first queries
> This is normal with Just-In-Time indexing. The system indexes articles as needed. For faster performance, build a full index (see above).

---

## Notes

```
┌──────────────────────────────────────────────────────────────┐
│ • Works without ZIM file, but no RAG capabilities           │
│ • First-time setup downloads ~1GB (Ollama + model + deps)   │
│ • GPU acceleration automatic if CUDA available               │
│ • All data stays local - no internet required after setup    │
└──────────────────────────────────────────────────────────────┘
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

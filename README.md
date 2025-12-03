# Kiwix RAG

Universal terminal chat app for Ollama with local Kiwix content integration. Features Retrieval Augmented Generation (RAG) for enhanced factual responses.

**Works with any ZIM file from the Kiwix library:**
- Wikipedia (any language)
- Wiktionary (dictionaries)
- Project Gutenberg (books)
- TED Talks
- And 100+ other educational content types

**⚠️ Platform Note:** For now this software is currently Linux-only. Windows and macOS support may be added in the future.

**⚠️ Disclaimer:** This software is provided "as is" without warranty. The author is not responsible for any misuse, damage, or consequences resulting from the use of this software. See [DISCLAIMER.md](DISCLAIMER.md) for full details.

## Features

- Terminal chat interface with streaming responses
- Automatic Kiwix content context injection for factual queries
- Works with ANY ZIM file from Kiwix library (Wikipedia, Wiktionary, Project Gutenberg, etc.)
- Clickable hyperlinks that open Kiwix content
- Multi-language support (any language ZIM file)
- Commands: `/help`, `/exit`, `/clear`, `/wiki <query>`, `/view <query>`

## Quick Start
Direct Download 

```bash
# 1. Download and extract the repository
wget https://github.com/imDelivered/OWRs/archive/refs/heads/main.zip
unzip main.zip
cd OWRs-main

# 2. Run the setup script
chmod +x setup.sh && ./setup.sh

# 3. Start the app (it handles everything automatically)
./run_wiki_chat.sh
```

**What happens automatically:**
- Setup script installs Python, Ollama, and Kiwix
- `run_wiki_chat.sh` starts Ollama server automatically
- Downloads the AI model if needed (first run only)
- Starts Kiwix server if ZIM file is found
- Launches the chat interface

**That's it!** No manual steps needed - just run `./run_wiki_chat.sh` after setup.

---

## Complete Setup Guide (Manual)

If you prefer to set up manually, follow these steps:

### Step 1: Install Python and Basic Tools

```bash
sudo apt update
sudo apt install -y python3 python3-tk curl wget
```

Verify Python is installed:
```bash
python3 --version
# Should show Python 3.10 or higher
```

### Step 2: Install Ollama

Ollama is the LLM server that runs the AI models.

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

Verify Ollama is installed:
```bash
ollama --version
```

### Step 3: Start Ollama and Download a Model

Start the Ollama server:
```bash
ollama serve
```

**Leave this terminal open.** Open a new terminal for the next steps.

In the new terminal, download a small model (llama3.2:1b is ~1.3GB, perfect for testing):
```bash
ollama pull llama3.2:1b
```

This will take a few minutes depending on your internet speed. You'll see download progress.

Verify the model downloaded:
```bash
ollama list
# Should show llama3.2:1b
```

### Step 4: Install Kiwix (for Local Content)

Kiwix serves ZIM files locally (Wikipedia, Wiktionary, Project Gutenberg, etc.).

```bash
sudo apt install -y kiwix-tools
```

Verify installation:
```bash
kiwix-serve --version
```

### Step 5: Download Kiwix ZIM File

The ZIM file contains educational content offline. This is optional but recommended. **The app works with ANY ZIM file from the Kiwix library** - not just Wikipedia!

**Browse and download from the official Kiwix Library:**
👉 **[https://library.kiwix.org/](https://library.kiwix.org/)**

**Universal Kiwix support:** The app works with any ZIM file from the Kiwix library:

**Wikipedia (any language):**
- **English**: `wikipedia_en_all_nopic_2025-07.zim` (~20GB)
- **Spanish**: `wikipedia_es_all_nopic_2025-07.zim` (~15GB)
- **French**: `wikipedia_fr_all_nopic_2025-07.zim` (~18GB)
- **German**: `wikipedia_de_all_nopic_2025-07.zim` (~20GB)
- **Simple English**: `wikipedia_en_simple_all_nopic_2025-07.zim` (~1.5GB, easier language)
- **Any other language** available in the Kiwix library

**Other content types:** The app works with ALL Kiwix ZIM files:
- **Wiktionary** - Dictionaries in multiple languages
- **Project Gutenberg** - Free ebooks
- **TED Talks** - Educational videos
- **Stack Overflow** - Programming Q&A
- **And 100+ other content types** available at [library.kiwix.org](https://library.kiwix.org)

Once downloaded, place the `.zim` file in your repo directory, `~`, or `/usr/share/kiwix/`. The app will auto-detect it.

**Or download directly via command line:**
```bash
# Navigate to your repo directory
cd /path/to/wiki-chat

# Download Wikipedia ZIM file (any language, no images)
# Example: English Wikipedia (~20GB)
wget -c https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_nopic_2025-07.zim

# Example: Spanish Wikipedia (~15GB)
wget -c https://download.kiwix.org/zim/wikipedia/wikipedia_es_all_nopic_2025-07.zim
```

**Note:** ZIM files can be large! You can skip this step if you just want to test the chat without Kiwix content features. The app will work, but `/wiki` commands won't function.

### Step 6: Get the Kiwix RAG Code

```bash
# Clone or download this repository
git clone <your-repo-url>
cd wiki-chat

# Or if you already have the files, just navigate to the directory
cd /path/to/wiki-chat
```

### Step 7: Run the App

Make the launcher executable:
```bash
chmod +x run_wiki_chat.sh
```

Run it:
```bash
./run_wiki_chat.sh
```

The launcher will:
- Check if Ollama is running (starts it if not)
- Check if your model is available (pulls it if missing)
- Start Kiwix server if ZIM file exists
- Launch the chat interface

**First time setup:** The launcher may take a minute to start everything.

### Step 8: Start Chatting!

Once the app launches, you'll see a chat interface. Just type your questions!

**Example:**
```
You: What is quantum computing?
AI: [Provides detailed answer with Wikipedia context if available]
```

## Usage

### Basic Commands

- `/help` - Show all available commands
- `/exit` - Quit the application
- `/clear` - Clear chat history
- `/wiki <query>` - Manually fetch Wikipedia articles (e.g., `/wiki Python programming`)
- `/view <query>` - Open Wikipedia article in popup window
- `/detailed on|off` - Toggle detailed response style
- `/links on|off` - Toggle hyperlink summary display

### Command-line Options

Run with custom settings:
```bash
python3 wiki_chat.py --model llama3.2:1b --detailed --wiki-max-chars 6000
```

Available options:
- `--model NAME` - Use different Ollama model (default: dolphin-llama3)
- `--detailed` - Start in detailed response mode
- `--wiki-max-chars N` - Max characters per Kiwix content item (default: 4000)
- `--zim-file PATH` - Specify path to ZIM file (any language/content type). If not specified, auto-detects first .zim file found.
- `--no-stream` - Disable streaming output
- `--no-links` - Disable hyperlink summary

## Troubleshooting

### Ollama Not Starting

If you see "Ollama not accessible":
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start it manually
ollama serve
```

### Model Not Found

If you see "model not found":
```bash
# List available models
ollama list

# Pull the model if missing
ollama pull llama3.2:1b
```

### Kiwix Not Working

If `/wiki` commands fail:
```bash
# Check if Kiwix is running
curl http://localhost:8081

# Check if ZIM file exists (any language)
ls -lh *.zim

# Start Kiwix manually (replace with your ZIM file path)
kiwix-serve --port=8081 /path/to/your/wikipedia_XX_all_nopic_2025-07.zim

# Or specify ZIM file when running the app
python3 wiki_chat.py --zim-file /path/to/your/wikipedia_es_all_nopic_2025-07.zim
```

### Python/Tkinter Issues

If GUI doesn't open:
```bash
# Install tkinter
sudo apt install -y python3-tk

# Test tkinter
python3 -c "import tkinter; print('OK')"
```

## System Requirements

**Platform:** Linux only (Windows and macOS support coming in the future)

**Minimum:**
- 4GB RAM (8GB recommended)
- 20GB free disk space (for model + ZIM file)
- Python 3.10+
- Internet connection (for initial setup)

**Recommended:**
- 16GB RAM
- 50GB free disk space
- 4+ CPU cores
- GPU (optional, for faster inference)

## What Each Component Does

- **Ollama**: Runs the AI language model (llama3, etc.)
- **Kiwix**: Serves Wikipedia articles from the ZIM file
- **ZIM File**: Contains all Wikipedia articles offline
- **wiki_chat.py**: The main chat application (Kiwix RAG)

## Notes

- The ZIM file is large (~20GB) and excluded from git
- Models are downloaded to `~/.ollama/models/` (several GB each)
- Works without ZIM file, but `/wiki` commands won't work
- Works with any Ollama model - change with `--model` flag
- First run may be slow as models load into memory

## Quick Reference

**Start everything:**
```bash
./run_wiki_chat.sh
```

**Manual start (if launcher has issues):**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Kiwix (if you have ZIM file, any language)
kiwix-serve --port=8081 /path/to/your/wikipedia_XX_all_nopic_2025-07.zim

# Terminal 3: Run the app
python3 wiki_chat.py --model llama3.2:1b
```

That's it! You're ready to chat with AI + Wikipedia.

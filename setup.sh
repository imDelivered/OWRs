#!/usr/bin/env bash
set -euo pipefail

echo "=== Kiwix RAG Setup Script ==="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Don't run this script as root. It will use sudo when needed."
    exit 1
fi

# Check sudo access
if ! sudo -n true 2>/dev/null; then
    echo "This script needs sudo access to install packages."
    echo "You may be prompted for your password."
    echo ""
fi

# Step 1: Update package list
echo "[1/6] Updating package list..."
sudo apt update -qq

# Step 2: Install Python and basic tools
echo "[2/6] Installing Python and basic tools..."
sudo apt install -y python3 python3-pip python3-tk curl wget > /dev/null 2>&1

# Verify Python
if ! python3 --version > /dev/null 2>&1; then
    echo "Error: Python3 installation failed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION installed"

# Step 3: Install Python dependencies
echo "[3/6] Installing Python dependencies..."
echo "  This may take a few minutes (downloading ~200MB of packages)..."
echo "  Installing: requests, sentence-transformers, chromadb, tiktoken"

# Function to verify installation
verify_install() {
    python3 -c "import chromadb; import sentence_transformers" 2>/dev/null
}

# Try --break-system-packages first (most reliable on modern systems)
INSTALL_SUCCESS=false
VERIFIED=false

echo "  Attempting installation with --break-system-packages..."
if python3 -m pip install --break-system-packages requests sentence-transformers chromadb tiktoken 2>&1; then
    if verify_install; then
        INSTALL_SUCCESS=true
        VERIFIED=true
        echo "  ✓ Installed and verified with --break-system-packages flag"
    else
        echo "  ⚠️  Installed but verification failed, trying --user method..."
    fi
fi

# If --break-system-packages didn't work, try --user
if [ "$VERIFIED" = false ]; then
    echo "  Attempting installation with --user flag..."
    if python3 -m pip install --user requests sentence-transformers chromadb tiktoken 2>&1; then
        if verify_install; then
            INSTALL_SUCCESS=true
            VERIFIED=true
            echo "  ✓ Installed and verified with --user flag"
        else
            # Sometimes --user installs but Python can't find them
            # Try adding user site-packages to path (find actual Python version)
            PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            USER_SITE="${HOME}/.local/lib/python${PYTHON_VERSION}/site-packages"
            if [ -d "$USER_SITE" ]; then
                export PYTHONPATH="${USER_SITE}:${PYTHONPATH}"
                if verify_install; then
                    INSTALL_SUCCESS=true
                    VERIFIED=true
                    echo "  ✓ Installed with --user flag (found in user site-packages)"
                fi
            fi
        fi
    fi
fi

if [ "$VERIFIED" = false ]; then
    echo ""
    echo "  ❌ ERROR: Failed to install or verify dependencies!"
    echo ""
    echo "  Please try installing manually:"
    echo "    pip3 install --break-system-packages sentence-transformers chromadb"
    echo ""
    echo "  If that fails, try:"
    echo "    pip3 install --user sentence-transformers chromadb"
    echo "    export PYTHONPATH=\${HOME}/.local/lib/python3.*/site-packages:\$PYTHONPATH"
    echo ""
    exit 1
fi

echo "✓ Python dependencies installed and verified (sentence-transformers, chromadb)"

# Step 4: Install Ollama
echo "[4/6] Installing Ollama..."
if ! command -v ollama > /dev/null 2>&1; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# Step 5: Install Kiwix
echo "[5/6] Installing Kiwix tools..."
if ! command -v kiwix-serve > /dev/null 2>&1; then
    sudo apt install -y kiwix-tools > /dev/null 2>&1
    echo "✓ Kiwix tools installed"
else
    echo "✓ Kiwix tools already installed"
fi

# Step 6: Make scripts executable
echo "[6/6] Setting up scripts..."
chmod +x run_kiwix_chat.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
echo "✓ Scripts made executable"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Everything is ready! Just run:"
echo ""
echo "  ./run_kiwix_chat.sh"
echo ""
echo "The launcher will automatically:"
echo "  • Start Ollama server"
echo "  • Download the AI model (if needed)"
echo "  • Start Kiwix server (if ZIM file found)"
echo "  • Launch the chat interface"
echo ""
echo "RAG System Setup:"
echo "  • Embedding models (BGE) are ready to use"
echo "  • To enable RAG: Download a ZIM file from https://library.kiwix.org/"
echo "  • Place the .zim file in this directory"
echo "  • Build the index: python3 kiwix_chat.py --build-index"
echo "  • This creates embeddings for semantic search (one-time, may take time)"
echo ""
echo "Optional: Download Wikipedia ZIM file from https://library.kiwix.org/"
echo "          Place it in this directory to enable Wikipedia features."


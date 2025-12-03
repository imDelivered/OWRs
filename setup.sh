#!/usr/bin/env bash
set -euo pipefail

echo "=== Wiki Chat Setup Script ==="
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
python3 -m pip install --user requests > /dev/null 2>&1 || true
echo "✓ Python dependencies installed"

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
chmod +x run_wiki_chat.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
echo "✓ Scripts made executable"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Everything is ready! Just run:"
echo ""
echo "  ./run_wiki_chat.sh"
echo ""
echo "The launcher will automatically:"
echo "  • Start Ollama server"
echo "  • Download the AI model (if needed)"
echo "  • Start Kiwix server (if ZIM file found)"
echo "  • Launch the chat interface"
echo ""
echo "Optional: Download Wikipedia ZIM file from https://library.kiwix.org/"
echo "          Place it in this directory to enable Wikipedia features."


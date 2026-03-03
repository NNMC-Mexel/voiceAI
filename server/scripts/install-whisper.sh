#!/bin/bash

# ============================================
# Whisper Installation Script for МедДок
# ============================================

echo "🎙️ Installing Whisper for Speech-to-Text..."

# Create models directory
mkdir -p models

# Option 1: whisper.cpp (Recommended for production - fastest)
install_whisper_cpp() {
    echo "📦 Installing whisper.cpp..."
    
    # Clone whisper.cpp
    git clone https://github.com/ggerganov/whisper.cpp.git
    cd whisper.cpp
    
    # Build with CUDA support (for NVIDIA GPUs)
    if command -v nvcc &> /dev/null; then
        echo "🎮 CUDA detected, building with GPU support..."
        make WHISPER_CUDA=1
    else
        echo "⚠️ CUDA not found, building CPU-only version..."
        make
    fi
    
    # Download large-v3 model (best quality for Russian)
    echo "📥 Downloading Whisper large-v3 model (~3GB)..."
    ./models/download-ggml-model.sh large-v3
    
    # Move model to our models directory
    cp models/ggml-large-v3.bin ../models/
    
    # Create symlink for easy access
    sudo ln -sf $(pwd)/main /usr/local/bin/whisper-cpp
    
    cd ..
    echo "✅ whisper.cpp installed successfully!"
}

# Option 2: faster-whisper (Python - good balance of speed and ease)
install_faster_whisper() {
    echo "📦 Installing faster-whisper (Python)..."
    
    pip install faster-whisper --break-system-packages
    
    # Pre-download the model
    python3 << 'EOF'
from faster_whisper import WhisperModel
print("📥 Downloading Whisper large-v3 model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("✅ Model downloaded and cached!")
EOF
    
    echo "✅ faster-whisper installed successfully!"
}

# Option 3: OpenAI Whisper (Original - slowest but most compatible)
install_openai_whisper() {
    echo "📦 Installing OpenAI Whisper (Python)..."
    
    pip install openai-whisper --break-system-packages
    
    # Pre-download the model
    python3 << 'EOF'
import whisper
print("📥 Downloading Whisper large-v3 model...")
model = whisper.load_model("large-v3")
print("✅ Model downloaded and cached!")
EOF
    
    echo "✅ OpenAI Whisper installed successfully!"
}

# Check system requirements
check_requirements() {
    echo "🔍 Checking system requirements..."
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "⚠️ No NVIDIA GPU detected. CPU mode will be slower."
    fi
    
    # Check RAM
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    echo "💾 Total RAM: ${total_ram}GB"
    
    if [ "$total_ram" -lt 16 ]; then
        echo "⚠️ Warning: Less than 16GB RAM. Large models may be slow."
    fi
    
    # Check disk space
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
    echo "💿 Available disk space: ${available_space}GB"
    
    if [ "$available_space" -lt 10 ]; then
        echo "❌ Error: Need at least 10GB free disk space."
        exit 1
    fi
}

# Main menu
echo ""
echo "Select Whisper installation method:"
echo "1) whisper.cpp (Recommended - Fastest, requires compilation)"
echo "2) faster-whisper (Python - Good balance)"
echo "3) OpenAI Whisper (Python - Most compatible)"
echo ""
read -p "Enter choice [1-3]: " choice

check_requirements

case $choice in
    1) install_whisper_cpp ;;
    2) install_faster_whisper ;;
    3) install_openai_whisper ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo ""
echo "🎉 Whisper installation complete!"
echo ""
echo "Configuration:"
echo "  Model path: ./models/ggml-large-v3.bin (for whisper.cpp)"
echo "  Language: ru (Russian)"
echo "  Device: cuda (if available)"

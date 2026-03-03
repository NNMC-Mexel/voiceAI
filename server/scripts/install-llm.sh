#!/bin/bash

# ============================================
# LLM Installation Script for МедДок
# Uses llama.cpp for local LLM inference
# ============================================

echo "🤖 Installing LLM (Qwen 2.5) for text structuring..."

# Create models directory
mkdir -p models

# Install llama.cpp
install_llama_cpp() {
    echo "📦 Installing llama.cpp..."
    
    # Clone llama.cpp
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    
    # Build with CUDA support
    if command -v nvcc &> /dev/null; then
        echo "🎮 CUDA detected, building with GPU support..."
        make LLAMA_CUDA=1
    else
        echo "⚠️ CUDA not found, building CPU-only version..."
        make
    fi
    
    # Build server
    make llama-server
    
    # Create symlinks
    sudo ln -sf $(pwd)/llama-server /usr/local/bin/llama-server
    sudo ln -sf $(pwd)/llama-cli /usr/local/bin/llama-cli
    
    cd ..
    echo "✅ llama.cpp installed successfully!"
}

# Download Qwen 2.5 model
download_qwen_model() {
    echo "📥 Downloading Qwen 2.5 model..."
    
    echo ""
    echo "Select model size (based on your VRAM):"
    echo "1) Qwen2.5-7B-Instruct (Q4_K_M) - ~4.5GB VRAM - Good for 8GB GPUs"
    echo "2) Qwen2.5-14B-Instruct (Q4_K_M) - ~8.5GB VRAM - Recommended for 12GB+ GPUs"
    echo "3) Qwen2.5-32B-Instruct (Q4_K_M) - ~19GB VRAM - Best quality, needs 24GB+ VRAM"
    echo ""
    read -p "Enter choice [1-3]: " model_choice
    
    case $model_choice in
        1)
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
            MODEL_NAME="qwen2.5-7b-instruct-q4_k_m.gguf"
            ;;
        2)
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"
            MODEL_NAME="qwen2.5-14b-instruct-q4_k_m.gguf"
            ;;
        3)
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf"
            MODEL_NAME="qwen2.5-32b-instruct-q4_k_m.gguf"
            ;;
        *)
            echo "Invalid choice, using 14B model"
            MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf"
            MODEL_NAME="qwen2.5-14b-instruct-q4_k_m.gguf"
            ;;
    esac
    
    echo "📥 Downloading ${MODEL_NAME}..."
    wget -c "${MODEL_URL}" -O "models/${MODEL_NAME}"
    
    # Create symlink for default model
    ln -sf "${MODEL_NAME}" models/qwen-instruct.gguf
    
    echo "✅ Model downloaded: models/${MODEL_NAME}"
}

# Create startup script
create_startup_script() {
    cat > start-llm-server.sh << 'SCRIPT'
#!/bin/bash

# LLM Server Startup Script
MODEL_PATH="${MODEL_PATH:-./models/qwen-instruct.gguf}"
PORT="${LLM_PORT:-8080}"
GPU_LAYERS="${GPU_LAYERS:-99}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"

echo "🚀 Starting LLM Server..."
echo "   Model: ${MODEL_PATH}"
echo "   Port: ${PORT}"
echo "   GPU Layers: ${GPU_LAYERS}"
echo "   Context Size: ${CONTEXT_SIZE}"

llama-server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --n-gpu-layers "${GPU_LAYERS}" \
    --ctx-size "${CONTEXT_SIZE}" \
    --threads $(nproc) \
    --parallel 2 \
    --cont-batching
SCRIPT
    
    chmod +x start-llm-server.sh
    echo "✅ Created start-llm-server.sh"
}

# Create systemd service (optional)
create_systemd_service() {
    echo ""
    read -p "Create systemd service for auto-start? [y/N]: " create_service
    
    if [[ "$create_service" =~ ^[Yy]$ ]]; then
        sudo tee /etc/systemd/system/meddok-llm.service > /dev/null << SERVICE
[Unit]
Description=MedDok LLM Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/start-llm-server.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE
        
        sudo systemctl daemon-reload
        echo "✅ Systemd service created: meddok-llm"
        echo "   Start: sudo systemctl start meddok-llm"
        echo "   Enable: sudo systemctl enable meddok-llm"
    fi
}

# Check GPU memory
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""
    fi
}

# Main installation
echo ""
check_gpu_memory
install_llama_cpp
download_qwen_model
create_startup_script
create_systemd_service

echo ""
echo "🎉 LLM installation complete!"
echo ""
echo "To start the LLM server:"
echo "  ./start-llm-server.sh"
echo ""
echo "Environment variables:"
echo "  MODEL_PATH  - Path to GGUF model (default: ./models/qwen-instruct.gguf)"
echo "  LLM_PORT    - Server port (default: 8080)"
echo "  GPU_LAYERS  - Number of layers to offload to GPU (default: 99)"
echo ""
echo "Test the server:"
echo "  curl http://localhost:8080/health"

#!/bin/bash
# Install FlashAttention on a GPU node
# This script should be run on a GPU node with CUDA/nvcc available

set -e

echo "=== FlashAttention Installation Script ==="
echo "This script installs FlashAttention in the uv environment"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.in" ]; then
    echo "Error: requirements.in not found. Please run from repo root."
    exit 1
fi

# Activate the uv environment
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found. Please create the uv environment first."
    exit 1
fi

source .venv/bin/activate

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed and in PATH."
    echo "You may need to:"
    echo "  1. Load a CUDA module: module load cuda/12.8"
    echo "  2. Or set CUDA_HOME: export CUDA_HOME=/path/to/cuda"
    exit 1
fi

# Check CUDA_HOME
if [ -z "$CUDA_HOME" ]; then
    # Try to infer CUDA_HOME from nvcc
    NVCC_PATH=$(which nvcc)
    CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
    export CUDA_HOME
    echo "Inferred CUDA_HOME: $CUDA_HOME"
else
    echo "Using CUDA_HOME: $CUDA_HOME"
fi

# Verify CUDA version
NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
echo "CUDA version: $NVCC_VERSION"

# Check PyTorch CUDA compatibility
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "PyTorch CUDA version: $PYTORCH_CUDA"

# Install FlashAttention
echo ""
echo "Installing FlashAttention..."
echo "This may take 10-20 minutes as it compiles CUDA kernels..."
echo ""

# Use uv pip with --no-build-isolation as recommended by FlashAttention docs
uv pip install flash-attn --no-build-isolation

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import flash_attn; print(f'FlashAttention version: {flash_attn.__version__}')" || {
    echo "Error: FlashAttention import failed"
    exit 1
}

echo ""
echo "✅ FlashAttention installed successfully!"
echo ""
echo "To verify it works, you can run:"
echo "  python -c \"import flash_attn; print(flash_attn.__version__)\""


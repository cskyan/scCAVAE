#!/bin/bash
# Enhanced launcher for scCAVAE training scripts with GPU selection

cd "$(dirname "$0")"

echo "=========================================="
echo "scCAVAE Training Launcher"
echo "=========================================="
echo ""

# Function to show GPU status
show_gpu_status() {
    echo "Current GPU Status:"
    echo "----------------------------------------"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s\n    Memory: %s/%s MB (%.1f%%)\n    Utilization: %s%%\n\n",
                     $1, $2, $3, $4, ($3/$4)*100, $5}'
    else
        echo "  nvidia-smi not found. Cannot display GPU status."
    fi
    echo "----------------------------------------"
}

# Show GPU status
show_gpu_status

# GPU selection
echo ""
echo "Select GPU:"
echo "  0) GPU 0"
echo "  1) GPU 1"
echo "  2) GPU 2"
echo "  a) Auto (use least utilized GPU)"
echo ""
read -p "Select GPU (0/1/2/a) [default: 0]: " gpu_choice
gpu_choice=${gpu_choice:-0}

# Determine which GPU to use
if [ "$gpu_choice" = "a" ] || [ "$gpu_choice" = "A" ]; then
    echo "Auto-selecting least utilized GPU..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
                 sort -t',' -k2 -n | head -1 | cut -d',' -f1)
        echo "Selected GPU $GPU_ID (least memory used)"
    else
        GPU_ID=0
        echo "nvidia-smi not found. Using GPU 0"
    fi
else
    GPU_ID=$gpu_choice
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"
echo ""

# Dataset selection
echo "Available datasets:"
echo "  1) Kang"
echo "  2) Norman"
echo "  3) Sciplex"
echo ""
read -p "Select dataset (1-3): " choice

# Reconstruction loss selection
echo ""
echo "Select reconstruction loss:"
echo "  1) MSE (Mean Squared Error) - Fast, simple"
echo "  2) NB (Negative Binomial) - Recommended for counts data"
echo "  3) ZINB (Zero-Inflated NB) - Best for high zero-rate data"
echo "  4) Gauss (Gaussian) - For normalized data"
echo "  5) Huber - Robust to outliers"
echo ""
read -p "Select loss (1-5) [default: 1-MSE]: " loss_choice
loss_choice=${loss_choice:-1}

case $loss_choice in
    1)
        export RECON_LOSS="mse"
        echo "Using MSE loss"
        ;;
    2)
        export RECON_LOSS="nb"
        echo "Using NB loss"
        ;;
    3)
        export RECON_LOSS="zinb"
        echo "Using ZINB loss"
        ;;
    4)
        export RECON_LOSS="gauss"
        echo "Using Gauss loss"
        ;;
    5)
        export RECON_LOSS="huber"
        echo "Using Huber loss"
        ;;
    *)
        export RECON_LOSS="mse"
        echo "Invalid choice. Using default MSE loss"
        ;;
esac
echo ""

case $choice in
    1)
        echo "Starting Kang training with $RECON_LOSS loss..."
        python train_kang.py
        ;;
    2)
        echo "Starting Norman training with $RECON_LOSS loss..."
        python train_norman.py
        ;;
    3)
        echo "Starting Sciplex training with $RECON_LOSS loss..."
        python train_sciplex.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

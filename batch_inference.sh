#!/bin/bash
"""
Batch inference script for running multiple folds
Usage: ./batch_inference.sh [dataset] [device]
Example: ./batch_inference.sh her2st cuda
"""

# Set default values
DATASET=${1:-her2st}
DEVICE=${2:-cuda}

echo "Running batch inference for dataset: $DATASET on device: $DEVICE"
echo "=================================================="

# Create results directory
mkdir -p results

# Run inference for each fold
for fold in {1..5}; do
    echo ""
    echo "Running inference for fold $fold..."
    echo "----------------------------------------"
    
    python run_inference.py \
        --fold $fold \
        --dataset $DATASET \
        --device $DEVICE \
        --save_predictions \
        --output_dir results
    
    if [ $? -eq 0 ]; then
        echo "Fold $fold completed successfully"
    else
        echo "Error in fold $fold"
        exit 1
    fi
done

echo ""
echo "All folds completed!"
echo "Results saved in ./results/"

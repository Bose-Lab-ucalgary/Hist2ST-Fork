# Hist2ST-Fork Inference Scripts

This directory contains scripts for running inference with the trained Hist2ST model.

## Files

- `run_inference.py` - Full-featured inference script with extensive options
- `simple_inference.py` - Simplified inference script for basic usage
- `batch_inference.sh` - Batch script for running inference on multiple folds
- `README_inference.md` - This documentation file

## Quick Start

### Basic Usage (Simple Script)

```bash
# Run inference for fold 5 on HER2ST dataset
python simple_inference.py

# Run inference for fold 3 on CSCC dataset
python simple_inference.py --fold 3 --dataset cscc

# Use CPU instead of GPU
python simple_inference.py --device cpu
```

### Advanced Usage (Full Script)

```bash
# Run inference with all options
python run_inference.py \
    --fold 5 \
    --dataset her2st \
    --device cuda \
    --save_predictions \
    --calculate_ari \
    --output_dir ./results

# Run inference and save results
python run_inference.py --fold 3 --save_predictions
```

### Batch Processing

```bash
# Make the script executable
chmod +x batch_inference.sh

# Run inference for all folds (1-5) on HER2ST dataset
./batch_inference.sh her2st cuda

# Run inference for all folds on CSCC dataset using CPU
./batch_inference.sh cscc cpu
```

## Script Parameters

### Common Parameters

- `--fold`: Fold number for cross-validation (default: 5)
- `--dataset`: Dataset type - 'her2st' or 'cscc' (default: 'her2st')
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')

### Advanced Parameters (run_inference.py only)

- `--model_dir`: Directory containing model checkpoints (default: './model')
- `--batch_size`: Batch size for inference (default: 1)
- `--tag`: Model configuration tag (default: '5-7-2-8-4-16-32')
- `--output_dir`: Directory to save results (default: './results')
- `--save_predictions`: Save prediction arrays to files
- `--calculate_ari`: Calculate ARI clustering score

## Model Configuration

The model uses the following hyperparameters (encoded in tag '5-7-2-8-4-16-32'):
- Kernel size: 5
- Patch size: 7
- Depth 1: 2
- Depth 2: 8
- Depth 3: 4
- Heads: 16
- Channels: 32

## Expected File Structure

```
Hist2ST-Fork/
├── model/
│   ├── 1-Hist2ST.ckpt
│   ├── 2-Hist2ST.ckpt
│   ├── 3-Hist2ST.ckpt
│   ├── 4-Hist2ST.ckpt
│   └── 5-Hist2ST.ckpt
├── predict.py
├── HIST2ST.py
├── dataset.py
├── run_inference.py
├── simple_inference.py
└── batch_inference.sh
```

## Output

The scripts will output:
1. Pearson correlation coefficient
2. ARI score (if requested and applicable)
3. Saved prediction arrays (if `--save_predictions` is used)
4. Metrics summary file

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `--device cpu` or reduce batch size
2. **Model checkpoint not found**: Check that model files exist in `./model/` directory
3. **Import errors**: Ensure all required Python files are in the same directory

### Example Error Solutions

```bash
# If GPU memory is insufficient
python simple_inference.py --device cpu

# If model checkpoint is in different location
python run_inference.py --model_dir /path/to/models

# If you want to use a specific fold
python simple_inference.py --fold 1
```

## Dependencies

Required Python packages:
- torch
- numpy
- pandas
- scipy
- scikit-learn
- tqdm

Make sure all model files (predict.py, HIST2ST.py, dataset.py) are available in the same directory.

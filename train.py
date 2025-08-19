"""
HEST Training Script
===================

This script provides a complete training pipeline for the Hist2ST model on the HEST dataset.
It supports configurable training parameters, multi-GPU training, automatic checkpointing,
and comprehensive monitoring with callbacks for memory management and time tracking.

Key Features:
- Flexible training modes (train, test, validate, train_test, all)
- Support for multiple datasets (HEST1K, HER2ST, SKIN)
- Configurable gene lists and model architectures
- Multi-GPU training with various strategies
- Automatic checkpoint resumption
- Memory optimization and monitoring
- Early stopping and model checkpointing

Usage:
    python HEST_train.py --mode train_test --dataset HEST1K --gene_list HER2ST --epochs 350


"""

import gc
import os
import torch
import random
import numpy as np
import argparse
import shutil
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datetime import date
from config import GENE_LISTS
# from custom_trainer import custom_eval_loop, custom_train_loop
from predict import test as predict
from callbacks import TimeTrackingCallback, GPUMemoryMonitorCallback, ClearCacheCallback, OOMHandlerCallback

# Import your modules here
from utils import *
from HIST2ST import *
from predict import *
from dataset import ViT_HEST1K, custom_collate_fn

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def parse_args():
    parser = argparse.ArgumentParser(description='Train and test Hist2ST model with configurable parameters')    
    # General parameters
    parser.add_argument('--mode', type=str, default='train_test', 
                        choices=['train', 'test', 'validate', 'train_test', 'all'],
                        help='Mode to run: train, test, validate, train_test (train then test), or all (train, validate, test)')
    parser.add_argument('--test_sample_id', type=int, default=0,
                        help='Test sample ID for naming')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='HEST1K',
                        choices=['HEST1K', 'HER2ST', 'SKIN'],
                        help='Dataset to use')
    parser.add_argument('--gene_list', type=str, default='HER2ST',
                        choices=list(GENE_LISTS.keys()),
                        help='Gene list to use')
    parser.add_argument('--cancer_only', type=bool, default=True,
                        help='Whether to use only cancer samples')
        # Hardware parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU only)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=[None, 'ddp', 'ddp_spawn', 'deepspeed'],
                        help='Training strategy for multi-GPU (None, ddp, ddp_spawn, deepspeed)')
    # Path parameters
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save/load models')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    parser.add_argument('--neighbors', type=int, default=5, help='Number of neighbors in GNN')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and testing')
    parser.add_argument('--precision', type=str, default='16', choices=['16', '32', 16, 32],
                        help='Training precision: "16" or "32" (string or int, as required by PyTorch Lightning)')
    parser.add_argument('--tag', type=str, default='5-7-1-4-2-8-16', 
                            help='Model hyperparameters in format: kernel-patch-depth1-depth2-depth3-heads-channel (e.g., "5-7-1-4-2-8-16" means kernel=5, patch=7, depth1=1, depth2=4, depth3=2, heads=8, channel=16)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                            help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--prune', type=str, default='NA', help='Pruning method (default: NA)')

    # Checkpoint management
    parser.add_argument('--auto_resume', action='store_true', default=True,
                        help='Automatically resume from last checkpoint if found (default: True)')
    parser.add_argument('--force_restart', action='store_true', default=False,
                        help='Force restart training, ignoring any existing checkpoints')
    parser.add_argument('--interactive', action='store_true', default=False,
                        help='Ask user whether to resume from found checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to load (overrides automatic path)')
    
    # Model-specific parameters
    parser.add_argument('--zinb', type=float, default=0.25, 
                        help='ZINB parameter for the model (default: 0.25)')
    parser.add_argument('--nb', action='store_true', default=False,
                        help='Enable NB parameter for the model (default: False)')
    parser.add_argument('--bake', type=float, default=0.0,
                        help='Bake parameter for the model (default: 0.0)')
    parser.add_argument('--lamb', type=float, default=0.0,
                        help='Lambda parameter for the model (default: 0.0)')
    
    return parser.parse_args()

def get_checkpoint_info(modelsave_address, args):
    """
    Check for existing checkpoints and determine what to load.
    This function searches for checkpoints in the following priority order:
    1. Explicit checkpoint path specified in args.checkpoint_path
    2. Automatic checkpoint (last.ckpt) in the model save directory
    3. Any .ckpt files in the model save directory (returns the most recently created)
    Args:
        modelsave_address (str): Directory path where model checkpoints are saved
        args: Arguments object containing checkpoint_path attribute for explicit checkpoint specification
    Returns:
        tuple: A tuple containing:
            - checkpoint_path (str or None): Path to the checkpoint file if found, None otherwise
            - checkpoint_type (str): Type of checkpoint found, one of:
                - "explicit": User-specified checkpoint path
                - "auto": Automatic last.ckpt checkpoint
                - "found": Latest .ckpt file in directory
                - "none": No checkpoints found
    Example:
        >>> checkpoint_path, checkpoint_type = get_checkpoint_info("/path/to/models", args)
        >>> if checkpoint_path:
        ...     print(f"Loading {checkpoint_type} checkpoint: {checkpoint_path}")
    """
    
    """Check for existing checkpoints and determine what to load"""
    # Explicit checkpoint path?
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Using explicit checkpoint: {args.checkpoint_path}")
        return args.checkpoint_path, "explicit"
    
    # auto-resume using last.ckpt
    last_checkpoint = os.path.join(modelsave_address, "last.ckpt")
    if os.path.exists(last_checkpoint):
        print(f"Found automatic checkpoint: {last_checkpoint}")
        return last_checkpoint, "auto"
    
    # any.ckpt files in directory?
    if os.path.exists(modelsave_address):
        ckpt_files = [f for f in os.listdir(modelsave_address) if f.endswith('.ckpt')]
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: os.path.getctime(os.path.join(modelsave_address, x)))
            latest_path = os.path.join(modelsave_address, latest_ckpt)
            print(f"Found existing checkpoint: {latest_path}")
            return latest_path, "found"
    
    print("No existing checkpoints found")
    return None, "none"

def should_resume_training(checkpoint_path, checkpoint_type, args):
    """
    Determine if training should resume from a checkpoint or start fresh.
    Args:
        checkpoint_path (str or None): Path to the checkpoint file to potentially resume from
        checkpoint_type (str): Type of checkpoint detection:
            - "none": No checkpoint found or specified
            - "explicit": Checkpoint explicitly provided by user
            - "auto": Automatically detected checkpoint from previous run
            - "found": Checkpoint found but requires user decision
        args (argparse.Namespace): Command line arguments containing:
            - force_restart (bool): If True, ignore existing checkpoints and start fresh
            - auto_resume (bool): If True, automatically resume from checkpoint
    Returns:
        tuple: (checkpoint_path, start_epoch)
            - checkpoint_path (str or None): Path to checkpoint to resume from, or None to start fresh
            - start_epoch (None): Always returns None for start epoch (handled elsewhere)
    Behavior:
        - "none" type: Always starts fresh training
        - "explicit" type: Always uses the provided checkpoint
        - Any type with force_restart=True: Starts fresh regardless of checkpoint
        - Any type with auto_resume=True: Resumes from checkpoint
        - "auto" type (default): Automatically resumes from checkpoint
        - "found" type (default): Starts fresh unless auto_resume is enabled
    """
    
    """Determine if training should resume or start fresh"""
    if checkpoint_type == "none":
        print("Starting fresh training...")
        return None, None
    
    if checkpoint_type == "explicit":
        print("Using explicitly provided checkpoint...")
        return checkpoint_path, None
    
    # For auto and found checkpoints, check user preference
    if args.force_restart:
        print("Force restart requested - ignoring existing checkpoints")
        return None, None
    
    if args.auto_resume:
        print("Auto-resuming from checkpoint...")
        return checkpoint_path, None
    
    # Interactive mode (if running interactively)
    # if hasattr(args, 'interactive') and args.interactive:
    #     response = input(f"Found checkpoint: {checkpoint_path}\nResume? (y/n): ").lower()
    #     if response.startswith('y'):
    #         return checkpoint_path, None
    #     else:
    #         print("Starting fresh training...")
    #         return None, None
    
    # Default: auto-resume for "auto" type, ask for "found" type
    if checkpoint_type == "auto":
        print("Auto-resuming from last checkpoint...")
        return checkpoint_path, None
    else:
        print("Found checkpoint but auto-resume not enabled. Starting fresh.")
        return None, None

def load_model_from_checkpoint(checkpoint_path, args):
    """
    Load a HisToGene model from a checkpoint file and extract dataset parameters.
    This function attempts to load a pre-trained model from a checkpoint and retrieve
    the dataset parameters that were used during training. It tries multiple methods
    to extract these parameters, falling back gracefully if they're not available.
    Args:
        checkpoint_path (str): Path to the model checkpoint file to load
        args: Command line arguments object containing fallback dataset parameters
              including gene_list, prune, neighbors, cancer_only, and dataset
    Returns:
        tuple: A tuple containing:
            - model (HisToGene or None): The loaded model instance, or None if loading failed
            - dataset_params (dict or None): Dictionary containing dataset parameters with keys:
                'gene_list', 'prune', 'neighbors', 'cancer_only', 'dataset_name'.
                Returns None if no parameters could be extracted from the checkpoint.
    Notes:
        - First attempts to use model's get_dataset_params() method if available
        - Falls back to extracting parameters from model.hparams if present
        - Uses command line args as fallback values when checkpoint params are missing
        - Prints status messages during the loading process
        - Returns (None, None) if checkpoint loading fails
    """
    
    """Load model and extract dataset parameters"""
    try:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = Hist2ST.load_from_checkpoint(checkpoint_path)
        
        # Check if model has dataset parameters
        if hasattr(model, 'get_dataset_params'):
            dataset_params = model.get_dataset_params()
            print(f"Model was trained with dataset params: {dataset_params}")
            return model, dataset_params
        elif hasattr(model, 'hparams'):
            # Extract dataset params from hyperparameters
            dataset_params = {
                'gene_list': getattr(model.hparams, 'gene_list', args.gene_list),
                'prune': getattr(model.hparams, 'prune', args.prune),
                'neighbors': getattr(model.hparams, 'neighbors', args.neighbors),
                'cancer_only': getattr(model.hparams, 'cancer_only', args.cancer_only),
                'dataset_name': getattr(model.hparams, 'dataset_name', args.dataset)
            }
            print(f"Extracted dataset params from hyperparameters: {dataset_params}")
            return model, dataset_params
        else:
            print("No dataset parameters found in checkpoint, using command line args")
            return model, None
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None
    
def train(args, vit_dataset=ViT_HEST1K):
    """
    Train a Hist2ST model on spatial transcriptomics data with comprehensive checkpoint management.
    This function handles the complete training pipeline including model creation/loading,
    dataset preparation, checkpoint management, and training execution with monitoring callbacks.
    Args:
        args: ArgumentParser namespace containing training configuration including:
            - model_dir (str): Directory to save model checkpoints
            - gene_list (str): Name of gene list to use from GENE_LISTS
            - prune (bool): Whether to prune the dataset
            - neighbors (int): Number of neighbors for spatial graph construction
            - cancer_only (bool): Whether to use only cancer samples
            - tag (str): Model architecture specification in format 'kernel-patch-depth1-depth2-depth3-heads-channel'
            - learning_rate (float): Learning rate for optimization
            - dropout (float): Dropout rate
            - zinb (bool): Whether to use zero-inflated negative binomial loss
            - nb (bool): Whether to use negative binomial loss
            - bake (bool): Baking parameter for model
            - lamb (float): Lambda parameter for regularization
            - dataset (str): Name of the dataset being used
            - num_workers (int): Number of data loading workers
            - epochs (int): Maximum number of training epochs
            - strategy (str): PyTorch Lightning training strategy
            - precision (str): Training precision (e.g., '16-mixed', '32')
            - checkpoint_dir (str): Directory to save final model
        vit_dataset (class, optional): Dataset class to use. Defaults to ViT_HEST1K.
    Returns:
        None: The function saves trained models and logs but doesn't return values.
    Raises:
        ValueError: If checkpoint directory cannot be created or is not writable.
    Notes:
        - Automatically resumes training from the latest checkpoint if available
        - Uses comprehensive monitoring including GPU memory, timing, and OOM handling
        - Saves model checkpoints every epoch and maintains the best model
        - Implements early stopping based on validation loss with 25 epoch patience
        - Creates CSV logs for training metrics tracking
        - Copies best model to final checkpoint directory upon completion
    """
    
    print('#'*50)
    print("CHECKPOINT MANAGEMENT:")
    modelsave_address = args.model_dir
    # checkpoint management
    checkpoint_path, checkpoint_type = get_checkpoint_info(modelsave_address, args)
    resume_path, loaded_model = should_resume_training(checkpoint_path, checkpoint_type, args)
    
    gene_list = args.gene_list
    prune = args.prune
    neighbors = args.neighbors
    cancer_only = args.cancer_only
    
    model = None

    # model creation/loading
    if resume_path:
        # Load existing model and dataset params
        loaded_model, dataset_params = load_model_from_checkpoint(resume_path, args)
        
        if loaded_model and dataset_params:
            # Use dataset parameters from checkpoint
            print("Using dataset parameters from checkpoint:")
            for key, value in dataset_params.items():
                print(f"  {key}: {value}")
            
            # Override args with checkpoint params for consistency
            gene_list = dataset_params.get('gene_list', gene_list)
            prune = dataset_params.get('prune', prune)
            neighbors = dataset_params.get('neighbors', neighbors)
            cancer_only = dataset_params.get('cancer_only', args.cancer_only)
            
            model = loaded_model
        else:
            print("!!Failed to load checkpoint, creating new model")
            resume_path = None
            
    if model is None:
        print("Creating new model...")
        kernel, patch, depth1, depth2, depth3, heads, channel = map(int, args.tag.split('-'))
        n_genes = GENE_LISTS[gene_list]["n_genes"]
        
        # Create model
        model = Hist2ST(
            depth1=depth1, depth2=depth2, depth3=depth3,
            n_genes=n_genes, learning_rate=args.learning_rate,
            kernel_size=kernel, patch_size=patch,
            heads=heads, channel=channel, dropout=args.dropout,
            zinb=args.zinb, nb=args.nb, bake=args.bake, lamb=args.lamb,
            # Add dataset parameters
            gene_list=gene_list,
            prune=prune,
            neighbors=neighbors,
            cancer_only=cancer_only,
            dataset_name=args.dataset
        )
        
    # Create datasets with the determined parameters
    trainset = vit_dataset(mode='train', flatten=False, adj=True, ori=True, 
                          prune=prune, neighs=neighbors, gene_list=gene_list, 
                          cancer_only=cancer_only)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=args.num_workers, 
                             shuffle=True, pin_memory=False, collate_fn=custom_collate_fn)

    valset = vit_dataset(mode='val', flatten=False, adj=True, ori=True, 
                        prune=prune, neighs=neighbors, gene_list=gene_list, 
                        cancer_only=cancer_only)
    val_loader = DataLoader(valset, batch_size=1, num_workers=args.num_workers, 
                           shuffle=False, pin_memory=False, collate_fn=custom_collate_fn)
    
    # Setup logging
    today = date.today().strftime("%Y-%m-%d")
    log_name = f'hist2st_hest1k_{today}'
    os.makedirs(modelsave_address, exist_ok=True)
    log_dir = os.path.join(modelsave_address, "../logs/")
    os.makedirs(log_dir, exist_ok=True)
    
    # Verify directory exists and is writable
    if not os.path.exists(modelsave_address):
        raise ValueError(f"Could not create checkpoint directory: {modelsave_address}")
    if not os.access(modelsave_address, os.W_OK):
        raise ValueError(f"Checkpoint directory not writable: {modelsave_address}")
    
    print(f"Saving checkpoints to: {os.path.abspath(modelsave_address)}")
    logger = CSVLogger(save_dir=log_dir,
                         name="my_test_log_" + log_name)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=modelsave_address,
        filename=f"{log_name}" + "_{epoch:02d}",
        monitor=None,
        save_top_k=-1,
        save_last=True,  # This creates 'last.ckpt'
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        # min_delta=0.00,
        patience=25,
        verbose=True,
        mode='min'
    )
    
    devices = torch.cuda.device_count() if torch.cuda.is_available() else None
    memory_monitor = GPUMemoryMonitorCallback(log_every_n_batches=20)
    cache_cleaner = ClearCacheCallback(clear_every_n_batches=2)
    oom_handler = OOMHandlerCallback()
    time_tracker = TimeTrackingCallback()
    
    # Setup trainer with configurable GPU settings
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'strategy': args.strategy,
        'logger': logger,
        'callbacks': [
            checkpoint_callback, 
            early_stop_callback,
            memory_monitor,
            cache_cleaner,
            oom_handler,
            time_tracker
        ],
        'check_val_every_n_epoch': 10,
        'enable_progress_bar': False,
        'log_every_n_steps': 50,
        'precision': args.precision,
        'gradient_clip_val': 0.5,
        'accumulate_grad_batches': 8,
        'detect_anomaly': False,  # Disable anomaly detection for memory
    }
    if devices is not None:
        trainer_kwargs['devices'] = devices

    trainer = pl.Trainer(**trainer_kwargs)
    
    # This will automatically resume if ckpt_path is provided
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)
    
    timing_stats = time_tracker.get_stats()
    if timing_stats:
        print(f"\nFinal timing statistics:")
        print(f"Average epoch time: {time_tracker._format_time(timing_stats['average_epoch_time'])}")
    
    
    # Load best model for evaluation
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        best_model = Hist2ST.load_from_checkpoint(best_model_path)
        
        # Save final model
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        final_save_path = os.path.join(args.checkpoint_dir, f"Hist2ST_HEST1k_final_{today}.ckpt")
        shutil.copy2(best_model_path, final_save_path)
        print(f"Final model saved to {final_save_path}")
    else:
        print("Warning: No best model checkpoint found")

if __name__ == "__main__":
    print(f"Script started on: {date.today().strftime('%Y-%m-%d')}")
    # Clear CUDA cache at start
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    args = parse_args()
        # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # CUDA setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Enable memory efficient settings
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Select dataset class
    dataset_classes = {
        'HEST1K': ViT_HEST1K,
        'HER2ST': ViT_HER2ST,
        'SKIN': ViT_SKIN
    }
    vit_dataset = dataset_classes[args.dataset]
    
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Gene List: {args.gene_list}")
    print(f"  Cancer Only: {args.cancer_only}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Directory: {args.model_dir}")
     
    # results = {}
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    train(
        args,
        vit_dataset=vit_dataset,        
    )
    # results['training'] = (pred_train, gt_train, R_train, p_val_train)
    # print(f"Training completed. Mean correlation: {np.nanmean(R_train):.4f}")
    
    # if len(results) > 1:
    #     print("\n" + "="*50)
    #     print("FINAL RESULTS SUMMARY")
    #     print("="*50)
        
    #     if 'training' in results:
    #         print(f"Training Correlation:   {np.nanmean(results['training'][2]):.4f}")
    #     if 'validation' in results:
    #         print(f"Validation Correlation: {np.nanmean(results['validation'][2]):.4f}")
    #     if 'testing' in results:
    #         print(f"Test Correlation:       {np.nanmean(results['testing'][2]):.4f}")

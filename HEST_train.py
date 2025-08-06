import gc
import os
import torch
import random
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datetime import date
from config import GENE_LISTS
from custom_trainer import custom_eval_loop, custom_train_loop
from predict import test as predict
from callbacks import TimeTrackingCallback, GPUMemoryMonitorCallback, ClearCacheCallback, OOMHandlerCallback

# Import your modules here
from utils import *
from HIST2ST import *
from predict import *

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
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to load (overrides automatic path)')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=350, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    parser.add_argument('--neighbors', type=int, default=5, help='Number of neighbors in GNN')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and testing')
    parser.add_argument('--precision', type=int, default=16, help='Training precision (16 or 32)')
    parser.add_argument('--tag', type=str, default='5-7-1-4-2-8-16', 
                            help='Hyperparameters: kernel-patch-depth1-depth2-depth3-heads-channel')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                            help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--prune', type=str, default='NA', help='Pruning method (default: NA)')
    return parser.parse_args()

def train(args, vit_dataset=ViT_HEST1K, epochs=300, modelsave_address="model", 
          gene_list="3CA", num_workers=16, gpus=1, strategy="ddp", learning_rate=1e-5, 
          batch_size=1, patience=25, tag='5-7-2-8-4-16-32', prune='NA', neighbors=5, 
          dropout=0.2, zinb=False, nb=False, bake=False, lamb=0.1):
    # Parse tag parameters
    kernel, patch, depth1, depth2, depth3, heads, channel = map(int, tag.split('-'))
    # Get number of genes from config
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    
    if gene_list not in GENE_LISTS:
        raise ValueError(f"Unknown gene list: {gene_list}")
    
    model = Hist2ST(
        depth1=depth1, depth2=depth2, depth3=depth3,
        n_genes=n_genes, learning_rate=learning_rate,
        kernel_size=kernel, patch_size=patch,
        heads=heads, channel=channel, dropout=dropout,
        zinb=zinb, nb=nb,
        bake=bake, lamb=lamb
    )

    # Load datasets
    trainset = vit_dataset(mode='train', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbors, gene_list=gene_list)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)

    valset = vit_dataset(mode='val', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbors, gene_list=gene_list)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)

    testset = vit_dataset(mode='test', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbors, gene_list=gene_list)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)

    # Get label for clustering evaluation if available
    # label = getattr(testset, 'label', {}).get(getattr(testset, 'names', [''])[0], None)
    
    # Setup logging
    today = date.today().strftime("%Y-%m-%d")
    log_name = f'hist2st_hest1k_{tag}_{today}'
    # logger = TensorBoardLogger('logs', name=log_name)
        # Setup logger
    logger = CSVLogger(save_dir=modelsave_address + "/../logs/",
                         name="my_test_log_" + log_name)
        
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=modelsave_address,
        filename=f"{log_name}" + "_{epoch:02d}_{valid_loss:.2f}",
        monitor='valid_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        # min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    # Configure trainer
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    memory_monitor = GPUMemoryMonitorCallback(log_every_n_batches=10)
    cache_cleaner = ClearCacheCallback(clear_every_n_batches=20)
    oom_handler = OOMHandlerCallback()
    time_tracker = TimeTrackingCallback()  # Add time tracking callback
    
    # Setup trainer with configurable GPU settings
    trainer_kwargs = {
        'max_epochs': epochs,
        'devices': devices,
        'strategy': strategy,
        'logger': logger,
        'callbacks': [
            checkpoint_callback, 
            early_stop_callback,
            memory_monitor,
            cache_cleaner,
            oom_handler,
            time_tracker  # Add to callbacks list
        ],
        'check_val_every_n_epoch': 5,
        'enable_progress_bar': False,
        'log_every_n_steps': 20,
        'precision': args.precision,
        'gradient_clip_val': 0.5,
        'accumulate_grad_batches': 16,
    }
    if gpus > 0:
        trainer_kwargs['accelerator'] = "gpu"
        if gpus == 1:
            trainer_kwargs['devices'] = [0]
        else:
            trainer_kwargs['devices'] = gpus
            if strategy:
                trainer_kwargs['strategy'] = strategy
    else:
        trainer_kwargs['accelerator'] = "cpu"
    
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)
    # Assuming you have a DataLoader called test_loader and a model
    
    # Define optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = torch.nn.MSELoss()  # or whatever loss function is appropriate
    # best_val_loss = float('inf')
    # patience_counter = 0
    
    # for epoch in range(epochs):
    #     custom_train_loop(model, train_loader, optimizer, loss_fn, device='cuda')
        
    #     # Validation step
    #     val_preds, val_gts, val_coords = custom_eval_loop(model, val_loader, device='cuda')
    #     val_loss = loss_fn(val_preds, val_gts).item()  # or another metric
    #     print(f"Epoch {epoch}: Validation loss = {val_loss:.4f}")
        
    #     # Save best model
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), "best_model.pt")
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1

    #     # Early stopping
    #     if patience_counter > patience:
    #         print("Early stopping triggered.")
    #         break
    
    timing_stats = time_tracker.get_stats()
    if timing_stats:
        print(f"\nFinal timing statistics:")
        print(f"Average epoch time: {time_tracker._format_time(timing_stats['average_epoch_time'])}")
    
    
    # Load best model for evaluation
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from {best_model_path}")
    best_model = Hist2ST.load_from_checkpoint(best_model_path)
    
    # Additional evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred, gt = pred(best_model, test_loader, device)
    R, p_val = get_R(pred, gt)[0]
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)
    print('Pearson Correlation:', np.nanmean(R))
    
    # if label is not None:
    #     clus, ARI = cluster(pred, label)
    #     print('ARI:', ARI)
    
    # Save final model
    final_save_path = os.path.join(args.checkpoint_dir, f"Hist2ST_HEST1k_final_{today}.ckpt")
    torch.save(best_model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    
    return pred, gt, R, p_val

def validate(vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", gene_list="3CA", checkpoint_path=None, num_workers=16, gpus=1, neighbours=5, prune="NA", batch_size=1):
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    # Load validation dataset
    valset = vit_dataset(mode='val', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbours, gene_list=gene_list)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # Load model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_address, f"last_train_{gene_list}_{dataset_name}_{n_genes}.ckpt")
    print(f"Loading model from: {checkpoint_path}")
    model = Hist2ST.load_from_checkpoint(checkpoint_path)
    # Run predictions
    device = 'cuda' if gpus > 0 and torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    pred, gt = test(model, val_loader, device)
    R, p_val = pred(pred, gt)[0]
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)
    print('Validation Pearson Correlation:', np.nanmean(R))
    return pred, gt, R, p_val

def run_test_phase(vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", gene_list="3CA", checkpoint_path=None, num_workers=16, gpus=1, neighbours=5, prune="NA", batch_size=1):
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    # Load test dataset
    testset = vit_dataset(mode='test', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbours, gene_list=gene_list)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    # Load model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_address, f"last_train_{gene_list}_{dataset_name}_{n_genes}.ckpt")
    print(f"Loading model from: {checkpoint_path}")
    model = Hist2ST.load_from_checkpoint(checkpoint_path)
    # Run predictions
    device = 'cuda' if gpus > 0 and torch.cuda.is_available() else 'cpu'
    pred, gt = test(model, test_loader, device)
    R, p_val = get_R(pred, gt)[0]
    if hasattr(pred, "var"):
        pred.var["p_val"] = p_val
        pred.var["-log10p_val"] = -np.log10(p_val)
    print('Test Pearson Correlation:', np.nanmean(R))
    return pred, gt, R, p_val

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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
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
    print(f"  GPUs: {args.gpus}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Directory: {args.model_dir}")
    
    
    results = {}
    
    if args.mode in ['train', 'train_test', 'all']:
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        pred_train, gt_train, R_train, p_val_train = train(
            args,
            vit_dataset=vit_dataset,
            epochs=args.epochs,
            modelsave_address=args.model_dir,
            gene_list=args.gene_list,
            num_workers=args.num_workers,
            gpus=args.gpus,
            strategy=args.strategy,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience,
            tag=args.tag,
            prune=args.prune,
            neighbors=args.neighbors,
            dropout=args.dropout
        )
        results['training'] = (pred_train, gt_train, R_train, p_val_train)
        print(f"Training completed. Mean correlation: {np.nanmean(R_train):.4f}")
        
    if args.mode in ['validate', 'all']:
        print("\n" + "="*50)
        print("VALIDATION PHASE")
        print("="*50)
        
        pred_val, gt_val, R_val, p_val_val = validate(
            vit_dataset=vit_dataset,
            model_address=args.model_dir,
            dataset_name=args.dataset.lower(),
            gene_list=args.gene_list,
            checkpoint_path=args.checkpoint_path,
            num_workers=args.num_workers,
            gpus=args.gpus
        )
        
        results['validation'] = (pred_val, gt_val, R_val, p_val_val)
        print(f"Validation completed. Mean correlation: {np.nanmean(R_val):.4f}")
    
    if args.mode in ['test', 'train_test', 'all']:
        print("\n" + "="*50)
        print("TESTING PHASE")
        print("="*50)
        
        pred_test, gt_test, R_test, p_val_test = run_test_phase(
            vit_dataset=vit_dataset,
            model_address=args.model_dir,
            dataset_name=args.dataset.lower(),
            gene_list=args.gene_list,
            checkpoint_path=args.checkpoint_path,
            num_workers=args.num_workers,
            gpus=args.gpus
        )
        
        results['testing'] = (pred_test, gt_test, R_test, p_val_test)
        print(f"Testing completed. Mean correlation: {np.nanmean(R_test):.4f}")
    
    # Print final summary if multiple phases were run
    if len(results) > 1:
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        
        if 'training' in results:
            print(f"Training Correlation:   {np.nanmean(results['training'][2]):.4f}")
        if 'validation' in results:
            print(f"Validation Correlation: {np.nanmean(results['validation'][2]):.4f}")
        if 'testing' in results:
            print(f"Test Correlation:       {np.nanmean(results['testing'][2]):.4f}")

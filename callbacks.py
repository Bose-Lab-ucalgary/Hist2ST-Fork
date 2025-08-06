import gc
import time
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch

class TimeTrackingCallback(Callback):
    """Track training time and calculate average time per epoch"""
    
    def __init__(self):
        self.epoch_start_times = []
        self.epoch_durations = []
        self.training_start_time = None
        self.total_training_time = 0
        
    def on_train_start(self, trainer, pl_module):
        self.training_start_time = time.time()
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        epoch_start = time.time()
        self.epoch_start_times.append(epoch_start)
        print(f"Epoch {trainer.current_epoch} started at: {time.strftime('%H:%M:%S', time.localtime(epoch_start))}")
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_times:
            epoch_end = time.time()
            epoch_duration = epoch_end - self.epoch_start_times[-1]
            self.epoch_durations.append(epoch_duration)
            
            # Calculate statistics
            avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
            
            # Format time
            epoch_time_str = self._format_time(epoch_duration)
            avg_time_str = self._format_time(avg_epoch_time)
            
            print(f"Epoch {trainer.current_epoch} completed in: {epoch_time_str}")
            print(f"Average epoch time: {avg_time_str}")
            
            # Estimate remaining time
            if trainer.max_epochs:
                remaining_epochs = trainer.max_epochs - trainer.current_epoch - 1
                estimated_remaining = remaining_epochs * avg_epoch_time
                if estimated_remaining > 0:
                    remaining_str = self._format_time(estimated_remaining)
                    eta = time.time() + estimated_remaining
                    eta_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eta))
                    print(f"Estimated remaining time: {remaining_str} (ETA: {eta_str})")
            
            print("-" * 50)
    
    def on_train_end(self, trainer, pl_module):
        if self.training_start_time:
            self.total_training_time = time.time() - self.training_start_time
            total_time_str = self._format_time(self.total_training_time)
            
            print("\n" + "="*50)
            print("TRAINING TIME SUMMARY")
            print("="*50)
            print(f"Total training time: {total_time_str}")
            
            if self.epoch_durations:
                avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
                min_epoch_time = min(self.epoch_durations)
                max_epoch_time = max(self.epoch_durations)
                
                print(f"Total epochs completed: {len(self.epoch_durations)}")
                print(f"Average time per epoch: {self._format_time(avg_epoch_time)}")
                print(f"Fastest epoch: {self._format_time(min_epoch_time)}")
                print(f"Slowest epoch: {self._format_time(max_epoch_time)}")
                
                # Calculate throughput
                if hasattr(trainer.datamodule, 'train_dataloader') or hasattr(trainer, 'train_dataloader'):
                    try:
                        # Try to get the number of batches
                        if hasattr(trainer, 'num_training_batches'):
                            num_batches = trainer.num_training_batches
                            avg_time_per_batch = avg_epoch_time / num_batches
                            print(f"Average time per batch: {avg_time_per_batch:.3f} seconds")
                    except:
                        pass
    
    def _format_time(self, seconds):
        """Format time in seconds to a readable string"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def get_stats(self):
        """Return timing statistics as a dictionary"""
        if not self.epoch_durations:
            return {}
        
        return {
            'total_training_time': self.total_training_time,
            'total_epochs': len(self.epoch_durations),
            'average_epoch_time': sum(self.epoch_durations) / len(self.epoch_durations),
            'min_epoch_time': min(self.epoch_durations),
            'max_epoch_time': max(self.epoch_durations),
            'epoch_durations': self.epoch_durations.copy()
        }
        

class GPUMemoryMonitorCallback(pl.Callback):
    """Monitor GPU memory usage during training"""
    
    def __init__(self, log_every_n_batches=10, device_id=0):
        self.log_every_n_batches = log_every_n_batches
        self.device_id = device_id
        
    def print_gpu_memory(self, prefix="", stage=""):
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(self.device_id) / 1024**3
            
            print(f"{stage} {prefix} GPU {self.device_id} Memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Max Allocated: {max_allocated:.2f} GB")
            print(f"  Max Reserved:  {max_reserved:.2f} GB")
    
    def on_train_start(self, trainer, pl_module):
        self.print_gpu_memory("Start", "TRAIN")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.print_gpu_memory(f"Epoch {trainer.current_epoch} Start", "TRAIN")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_batches == 0:
            self.print_gpu_memory(f"Batch {batch_idx}", "TRAIN")
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.print_gpu_memory(f"Epoch {trainer.current_epoch} End", "TRAIN")
    
    def on_validation_start(self, trainer, pl_module):
        self.print_gpu_memory("Start", "VAL")
    
    def on_validation_end(self, trainer, pl_module):
        self.print_gpu_memory("End", "VAL")

class ClearCacheCallback(pl.Callback):
    """Clear GPU cache at strategic points"""
    
    def __init__(self, clear_every_n_batches=20):
        self.clear_every_n_batches = clear_every_n_batches
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.clear_every_n_batches == 0:
            self.clear_cache()
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.clear_cache()
    
    def on_validation_end(self, trainer, pl_module):
        self.clear_cache()

class OOMHandlerCallback(pl.Callback):
    """Handle out-of-memory errors during training"""
    
    def __init__(self):
        self.oom_count = 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # This won't catch OOM during forward pass, but can help with cleanup
        pass
    
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception):
            self.oom_count += 1
            print(f"OOM Error #{self.oom_count} detected: {exception}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            print("Cleared CUDA cache after OOM")
            
            # Log memory state
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Post-OOM Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

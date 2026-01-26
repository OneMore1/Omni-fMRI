import torch
import torch.distributed as dist
import os
import numpy as np

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    world_size = dist.get_world_size()
    
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class LabelScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, labels):
        return (labels - self.mean) / self.std
    

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler=None):
    """Load checkpoint"""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0, 0.0

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    start_epoch = checkpoint['epoch']
    best_metric = checkpoint.get('best_metric', 0.0)
    best_loss = checkpoint.get('best_loss', float('inf'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"Loaded checkpoint from epoch {start_epoch}")
    return start_epoch, best_metric, best_loss


def set_seed(seed, rank=0):
    """Set random seed for reproducibility"""
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

import os
import sys
import argparse
import yaml
import time
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from functools import partial


from src.models.jepa_model import AdaptiveJEPA
from src.models.mae_model import AdaptiveMAE
from src.models.patch_embed_3d import TokenizedZeroConvPatchAttn3D
from src.data.pretrain_dataset import fMRIDataset
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def log_to_file(log_file, message):
    """Write message to log file"""
    if log_file is not None:
        with open(log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
            f.flush()


def count_parameters(model, verbose=True):

    def format_number(num):
        if num >= 1e9: return f"{num/1e9:.2f}B"
        elif num >= 1e6: return f"{num/1e6:.2f}M"
        elif num >= 1e3: return f"{num/1e3:.2f}K"
        else: return str(num)

    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, 'backbone'): 
        model = model.backbone

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Model Architecture: {type(model).__name__}")
        
        for name, child in model.named_children():
            num_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
            
            if num_params > 0:
                percentage = (num_params / total_params) * 100
                print(f"{name:.<35} {num_params:>15,} ({format_number(num_params):>8}) | {percentage:>6.2f}%")
                
                if "block" in name.lower() or "layer" in name.lower():
                    if isinstance(child, (nn.ModuleList, nn.Sequential)) and len(child) > 0:
                        print(f"  > Contains {len(child)} sub-blocks.")
                        first_block_params = sum(p.numel() for p in child[0].parameters() if p.requires_grad)
                        print(f"  > Approx params per block: {format_number(first_block_params)}")

    if verbose:
        print("\n" + "="*80)
        print(f"{'TOTAL TRAINABLE PARAMETERS':.<35} {total_params:>15,} ({format_number(total_params):>8})")
        print("="*80 + "\n")
    
    return {'total': total_params}


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30)
    )
    dist.barrier()
    return True, rank, world_size, gpu


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank=0):
    """Set random seed for reproducibility"""
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(config):
    model_config = config['model']

    if model_config['model_chose'] == 'mae':
        model = AdaptiveMAE(
            img_size=tuple(model_config['img_size']),
            patch_size=model_config['patch_size'],
            in_chans=model_config['in_chans'],
            embed_dim=model_config['embed_dim'],
            depth=model_config['depth'],
            qkv_bias=model_config['qkv_bias'],
            qk_norm=model_config['qk_norm'],
            num_heads=model_config['num_heads'],
            decoder_embed_dim=model_config['decoder_embed_dim'],
            drop_path_rate=model_config['drop_path_rate'],
            decoder_depth=model_config['decoder_depth'],
            decoder_num_heads=model_config['decoder_num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            mask_ratio=model_config['mask_ratio'],
            mixed_patch_embed=TokenizedZeroConvPatchAttn3D,
            patch_norm=model_config['enable_patch_norm'],
            gate_attention=model_config['gate_attention']
        )
        
    elif model_config['model_chose'] == 'jepa':
        model = AdaptiveJEPA(
            img_size=tuple(model_config['img_size']),
            patch_size=model_config['patch_size'],
            in_chans=model_config['in_chans'],
            embed_dim=model_config['embed_dim'],
            depth=model_config['depth'],
            qkv_bias=model_config['qkv_bias'],
            qk_norm=model_config['qk_norm'],
            num_heads=model_config['num_heads'],
            mlp_ratio=model_config['mlp_ratio'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            mask_ratio=model_config['mask_ratio'],
            mixed_patch_embed=TokenizedZeroConvPatchAttn3D,
            gate_attention=model_config['gate_attention'],
            lamda=model_config['lamda'],
            num_slices=model_config['num_slices'],
            n_points=model_config['n_points'],
            proj_dim=model_config['proj_dim'],
            predictor_embed_dim=model_config['predictor_embed_dim'],
            predictor_depth=model_config['predictor_depth'],
            use_patch_loss=model_config['use_patch_loss'],
            use_flatten_tokens=model_config['use_flatten_tokens'],
            sample_ratio=model_config['sample_ratio'],
            predictor_num_heads=model_config['predictor_num_heads']
        )

    return model


def create_optimizer(model, config):
    """Create optimizer from config"""
    train_config = config['training']

    if train_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            betas=tuple(train_config['betas']),
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config.get('momentum', 0.9),  
            weight_decay=train_config.get('weight_decay', 0.0),
            nesterov=train_config.get('nesterov', False) 
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")

    return optimizer


def create_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    train_config = config['training']
    total_steps = train_config['epochs'] * steps_per_epoch
    warmup_steps = train_config['warmup_epochs'] * steps_per_epoch

    if train_config['lr_scheduler'].lower() == 'cosine':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(train_config['min_lr'] / train_config['learning_rate'],
                          0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {train_config['lr_scheduler']}")

    return scheduler


def create_dataloaders(config, is_distributed, rank, world_size):
    """Create train and validation dataloaders"""
    data_config = config['data']

    # Training dataset
    train_dataset = fMRIDataset(
        data_root=data_config['data_root'],
        datasets=data_config['datasets'],
        split_suffixes=data_config['train_split_suffixes'],
        crop_length=data_config['input_seq_len'],
        downstream=config['model']['downstream']
    )

    # Validation dataset
    val_dataset = fMRIDataset(
        data_root=data_config['data_root'],
        datasets=data_config['datasets'],
        split_suffixes=data_config['val_split_suffixes'],
        crop_length=data_config['input_seq_len'],
        downstream=config['model']['downstream']
    )

    # Create samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['experiment']['seed']
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config.get('prefetch_factor', 2),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config.get('prefetch_factor', 2),
        drop_last=False
    )

    return train_loader, val_loader, train_sampler


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
        return 0, float('inf')

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('best_loss', float('inf'))

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    print(f"Loaded checkpoint from epoch {start_epoch}")
    return start_epoch, best_loss



def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, config,
                    rank, world_size, log_file=None):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    train_config = config['training']
    log_config = config['logging']

    accum_iter = train_config['accum_iter']
    use_amp = train_config['use_amp']
    clip_grad = train_config.get('clip_grad', 1.0)

    optimizer.zero_grad()
    
    for data_iter_step, samples in enumerate(metric_logger.log_every(train_loader, log_config['print_freq'], header)):

        # Adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            scheduler.step()

        # current_step = epoch * len(train_loader) + data_iter_step
        if torch.cuda.is_available():
            if isinstance(samples, (list, tuple)):
                samples = [s.cuda(rank, non_blocking=True) for s in samples]
            else:
                # Move data to GPU
                samples = samples.cuda(rank, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            if config['model']['model_chose'] == 'jepa':
                loss, sim_loss, sigreg_loss, aliment_metric = model(samples)
            elif config['model']['model_chose'] == 'mae':
                loss, rec_loss, aux_loss = model(samples)
            loss = loss / accum_iter    

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()

            if (data_iter_step + 1) % accum_iter == 0:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
        
            if (data_iter_step + 1) % accum_iter == 0:
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                optimizer.zero_grad()
        # Synchronize loss across GPUs

        loss_value = loss.item() * accum_iter
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        if config['model']['model_chose'] == 'jepa':
            metric_logger.update(loss=loss.item())
            metric_logger.update(sim_loss=sim_loss.item())
            metric_logger.update(sigreg_loss=sigreg_loss.item())
            metric_logger.update(aliment_metric=aliment_metric.item())
        elif config['model']['model_chose'] == 'mae':
            metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if current_step % 100 == 0:
        #     visual_g = g_output.clone()
        #     tensor_to_gather = visual_g.detach()

        #     gathered_tensors = [torch.zeros_like(tensor_to_gather) for _ in range(world_size)]
        #     dist.all_gather(gathered_tensors, tensor_to_gather)

        #     if rank == 0:
        #         visual_g = torch.cat(gathered_tensors, dim=0)
        #         visualize_local_similarity(visual_g, rank, epoch, step=current_step, save_dir="./vis/lejepa")

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, val_loader, epoch, config, rank, world_size, log_file=None):
    """Validate the model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Validation Epoch: [{epoch}]'


    for samples in metric_logger.log_every(val_loader, 50, header):
        # Move data to GPU
        if torch.cuda.is_available():
            if isinstance(samples, (list, tuple)):
                samples = [s.cuda(rank, non_blocking=True) for s in samples]
            else:
                # Move data to GPU
                samples = samples.cuda(rank, non_blocking=True)
                
        # Forward pass
        with autocast(enabled=True):
            if config['model']['model_chose'] == 'jepa':
                loss, sim_loss, sigreg_loss, aliment_metric = model(samples)
            elif config['model']['model_chose'] == 'mae':
                loss, _, _ = model(samples)

        if config['model']['model_chose'] == 'jepa':
            metric_logger.update(loss=loss.item())
            metric_logger.update(sim_loss=sim_loss.item())
            metric_logger.update(sigreg_loss=sigreg_loss.item())
            metric_logger.update(aliment_metric=aliment_metric.item())
        elif config['model']['model_chose'] == 'mae':
            metric_logger.update(loss=loss.item())


    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Validation averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class MetricLogger:
    """Metric logger for training"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available() and dist.get_rank() == 0:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')


class SmoothedValue:
    """Track a series of values and provide access to smoothed values"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.window_size = window_size

    def update(self, value, n=1):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize across all processes"""
        if not dist.is_available() or not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = sorted(self.deque)
        n = len(d)
        if n == 0:
            return 0
        if n % 2 == 0:
            return (d[n // 2 - 1] + d[n // 2]) / 2
        return d[n // 2]

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=max(self.deque) if len(self.deque) > 0 else 0,
            value=self.deque[-1] if len(self.deque) > 0 else 0
        )


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='fMRI Pretraining')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_val', type=bool, default=False, help='Disable validation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.resume is not None:
        config['experiment']['resume'] = args.resume
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir

    # Setup distributed training
    is_distributed, rank, world_size, gpu = setup_distributed()

    # Set random seed
    set_seed(config['experiment']['seed'], rank)

    # Create output directories
    if rank == 0:
        output_dir = Path(config['experiment']['output_dir'])
        checkpoint_dir = output_dir / 'checkpoints'
        log_dir = output_dir / 'logs'

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Setup text log file
        log_file = output_dir / 'training_log.txt'
        with open(log_file, 'w') as f:
            f.write(f"Training started at {datetime.datetime.now()}\n")
            f.write("="*80 + "\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Output directory: {config['experiment']['output_dir']}\n")
            f.write("="*80 + "\n\n")
    else:
        checkpoint_dir = None
        log_file = None

    if is_distributed:
        dist.barrier()

    # Print configuration
    if rank == 0:
        print(f"Config: {args.config}")
        print(f"Output directory: {config['experiment']['output_dir']}")
        print(f"Distributed: {is_distributed}")
        if is_distributed:
            print(f"World size: {world_size}")
            print(f"Rank: {rank}")
        print("="*80)

    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(config)
    model = model.cuda(gpu)

    # Count and print model parameters before wrapping with DDP
    if rank == 0:
        print("\nAnalyzing model architecture...")
        param_stats = count_parameters(model, verbose=True)

    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    model_without_ddp = model.module if is_distributed else model

    # Create dataloaders
    if rank == 0:
        print("Creating dataloaders...")
    train_loader, val_loader, train_sampler = create_dataloaders(
        config, is_distributed, rank, world_size
    )

    if rank == 0:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batches per epoch: {len(train_loader)}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model_without_ddp, config)
    scheduler = create_lr_scheduler(optimizer, config, len(train_loader))

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    if config['experiment']['resume'] is not None:
        start_epoch, best_loss = load_checkpoint(
            config['experiment']['resume'],
            model_without_ddp,
            optimizer,
            scheduler,
            scaler
        )

    # Training loop
    if rank == 0:
        print("Starting training...")
        print(f"Training from epoch {start_epoch} to {config['training']['epochs']}")

    start_time = time.time()

    for epoch in range(start_epoch, config['training']['epochs']):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, config, rank, world_size, log_file
        )

        is_val_epoch = not args.no_val and \
                       (epoch % config['validation']['val_freq'] == 0 or 
                        epoch == config['training']['epochs'] - 1)

        # Validate
        val_stats = {}
        if is_val_epoch:
            if len(val_loader) > 0:
                val_stats = validate(
                    model, val_loader, epoch, config, rank, world_size, log_file
                )
            else:
                if rank == 0:
                    print(f"Warning: Validation skipped because val_loader is empty.")

        if rank == 0:
            log_msg = f"Epoch {epoch} Training - " + " | ".join([f"{k}: {v:.4f}" for k, v in train_stats.items()])
            print(log_msg)
            log_to_file(log_file, log_msg)

            is_best = False
            if 'loss' in val_stats:
                val_loss = val_stats['loss']
                val_msg = f"Epoch {epoch} Validation - Loss: {val_loss:.4f}"
                print(val_msg)
                log_to_file(log_file, val_msg)

                if val_loss < best_loss:
                    best_loss = val_loss
                    is_best = True
                    best_msg = f"--> New best validation loss: {best_loss:.4f}"
                    print(best_msg)
                    log_to_file(log_file, best_msg)

            should_save_periodic = (epoch + 1) % config['logging']['save_freq'] == 0
            should_save_best = is_best

            if should_save_periodic or should_save_best:
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'config': config,
                }
                if scaler is not None:
                    checkpoint_state['scaler_state_dict'] = scaler.state_dict()

                if should_save_periodic:
                    save_checkpoint(
                        checkpoint_state,
                        False, 
                        checkpoint_dir,
                        filename=f'checkpoint_epoch_{epoch}.pth'
                    )
                    print(f"Saved periodic checkpoint: checkpoint_epoch_{epoch}.pth")

                if should_save_best:
                    save_checkpoint(
                        checkpoint_state,
                        True,
                        checkpoint_dir,
                        filename='best_checkpoint.pth' 
                    )
                    print(f"Saved best checkpoint: best_checkpoint.pth")
    # Training finished
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if rank == 0:
        summary = [
            "="*80,
            f"Training completed in {total_time_str}",
            f"Best validation loss: {best_loss:.4f}",
            f"Total epochs: {config['training']['epochs']}",
            f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}",
            "="*80
        ]

        for line in summary:
            print(line)
            log_to_file(log_file, line)

    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()


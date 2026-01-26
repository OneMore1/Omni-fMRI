import os
import sys
import argparse
import yaml
import time
import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from src.models.vision_transformer import VisionTransformer 
from src.models.patch_embed_3d import TokenizedZeroConvPatchAttn3D
from src.data.downstream_dataset import fMRITaskDataset, fMRITaskDataset1, EmoFMRIDataset, HCPtaskDataset
from src.data.adni_dataset import ADNIDataset

from src.utils.logging_utils import MetricLogger, log_to_file, load_config, count_parameters
from src.utils.dist_ddp import setup_distributed, cleanup_distributed
from src.utils.optim import create_lr_scheduler, create_optimizer
from src.utils.utils import LabelScaler, save_checkpoint, load_checkpoint, set_seed


def create_model(config):
    task_config = config['task']
    exp_config = config['experiment']

    # Determine model config
    model_config = config['model']
    pretrained_checkpoint_path = exp_config.get('pretrained_checkpoint', None)

    model = VisionTransformer(
        img_size=tuple(model_config['img_size']),
        patch_size=model_config['patch_size'],
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        qkv_bias=model_config['qkv_bias'],
        norm_layer=nn.LayerNorm,
        global_pool=model_config['global_pool'],
        fusion_mode=model_config['fusion_mode'],
        num_classes=task_config['num_classes'],
        downstream=model_config['downstream'],
        in_chans=model_config['in_chans'],
        gate_attention=model_config['gate_attention'],
        num_scales=model_config['num_scales'],
        method=model_config['method'],
        thresholds=model_config['thresholds'],
        freeze_backbone=config['training']['freeze_encoder'],
        mixed_patch_embed=TokenizedZeroConvPatchAttn3D 
    )

    if pretrained_checkpoint_path:
        print(f"Loading pretrained weights from: {pretrained_checkpoint_path}")
        if not os.path.isfile(pretrained_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {pretrained_checkpoint_path}")
            
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        
        # 1. Get state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('backbone.'):
                k = k[9:]
            if k.startswith('encoder.'):
                k = k[8:]
            new_state_dict[k] = v
        
        keys_to_remove = []
        for k in new_state_dict.keys():
            if k.startswith('head'):
                keys_to_remove.append(k)
        
        if keys_to_remove:
            print(f"Removing pretrained head weights (will be initialized randomly): {keys_to_remove}")
            for k in keys_to_remove:
                del new_state_dict[k]

        # 5. Load Weights
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Pretrained weights loaded.")
        print(f"Missing keys (initialized randomly): {len(msg.missing_keys)}")
        if len(msg.missing_keys) < 20:
            print(f"Missing: {msg.missing_keys}")
        if len(msg.missing_keys) > 0:
            print("List of missing keys:")
            for key in msg.missing_keys:
                print(f"  - {key}")
        print(f"Unexpected keys (ignored): {len(msg.unexpected_keys)}")

    else:
        print("No pretrained checkpoint specified. Initializing model with random weights.")

    return model


def create_dataloaders(config, is_distributed, rank, world_size):
    """Create train, validation, and test dataloaders"""
    data_config = config['data']
    task_config = config['task']

    if data_config['mode'] == "directory":
        train_dataset = fMRITaskDataset(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            split_suffixes=data_config['train_split_suffixes'],
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )

        val_dataset = fMRITaskDataset(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            split_suffixes=data_config['val_split_suffixes'],
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )


        test_dataset = fMRITaskDataset(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            split_suffixes=data_config.get('test_split_suffixes', ['test']),
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )
    
    elif data_config['mode'] == "txt":
        train_dataset = fMRITaskDataset1(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            split_suffixes=data_config['train_split_suffixes'],
            subject_list_txt=data_config['train_txt'],
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )

        val_dataset = fMRITaskDataset1(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            split_suffixes=data_config['val_split_suffixes'],
            subject_list_txt=data_config['val_txt'],
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )

        test_dataset = fMRITaskDataset1(
            data_root=data_config['data_root'],
            datasets=data_config['datasets'],
            subject_list_txt=data_config['test_txt'],
            split_suffixes=data_config.get('test_split_suffixes', ['test']),
            crop_length=data_config['input_seq_len'],
            label_csv_path=task_config['csv'],
            task_type=task_config['task_type']
        )

    elif data_config['mode'] == "emo":

        train_dataset = EmoFMRIDataset(
            txt_file="",
            csv_dir=""
        )

        val_dataset = EmoFMRIDataset(
            txt_file="",
            csv_dir=""
        )

        test_dataset = EmoFMRIDataset(
            txt_file="",
            csv_dir=""
        )

    elif data_config['mode'] == "task":

        train_dataset = HCPtaskDataset(
            txt_path=""
        )

        val_dataset = HCPtaskDataset(
            txt_path=""
        )

        test_dataset = HCPtaskDataset(
            txt_path=""
        )

    elif data_config['mode'] == "adni":
        train_dataset = ADNIDataset(txt_file="")
        val_dataset = ADNIDataset(txt_file="")
        test_dataset = ADNIDataset(txt_file="")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['experiment']['seed']
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        sampler=test_sampler,
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        prefetch_factor=data_config.get('prefetch_factor', 2),
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_sampler


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config,
                    rank, label_scaler=None):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    train_config = config['training']
    log_config = config['logging']
    task_config = config['task']

    accum_iter = train_config['accum_iter']
    use_amp = train_config['use_amp']
    clip_grad = train_config.get('clip_grad', None)

    optimizer.zero_grad()

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(train_loader, log_config['print_freq'], header)):
        # Adjust learning rate per iteration
        if data_iter_step % accum_iter == 0:
            scheduler.step()

        # Move data to GPU
        samples = samples.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)


        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(samples)

            # Calculate loss based on task type
            if task_config['task_type'] == 'classification':
                if labels.dim() > 1:
                    labels = labels.squeeze()

                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                accuracy = correct / labels.size(0)
            else:  
                if label_scaler is not None:
                    target_for_loss = label_scaler.transform(labels)
                else:
                    target_for_loss = labels
                loss = criterion(outputs.squeeze(), target_for_loss.squeeze())
                accuracy = 0.0 

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

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if task_config['task_type'] == 'classification':
            metric_logger.update(acc=accuracy)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, criterion, config, rank, epoch=None, label_scaler=None, mode='val'):

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'{mode.capitalize()} Epoch: [{epoch}]' if epoch is not None else f'{mode.capitalize()}:'
    
    task_type = config['task']['task_type']

    all_preds, all_targets = [], []

    for samples, labels in metric_logger.log_every(data_loader, 50, header):
        samples = samples.cuda(rank, non_blocking=True)
        labels = labels.cuda(rank, non_blocking=True)

        outputs = model(samples)

        if task_type == 'classification':
            labels = labels.squeeze().long() if labels.dim() > 1 else labels.long()
            loss = criterion(outputs, labels)
            
            preds = outputs.argmax(1)
            acc = (preds == labels).float().mean().item()
            metric_logger.update(loss=loss.item(), acc=acc)
            
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            
        else:  # regression
            target_norm = label_scaler.transform(labels) if label_scaler else labels
            loss = criterion(outputs.view(-1), target_norm.view(-1))
            
            metric_logger.update(loss=loss.item())
            all_preds.append(outputs.detach().cpu().view(-1))
            all_targets.append(target_norm.detach().cpu().view(-1))

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        if task_type == 'classification':
            f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average='weighted')
            metric_logger.update(f1=f1)
        else:
            mse = torch.mean((all_preds - all_targets) ** 2).item()
            mae = torch.mean(torch.abs(all_preds - all_targets)).item()
            
            ss_res = torch.sum((all_targets - all_preds) ** 2)
            ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)
            r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
            
            vx = all_preds - all_preds.mean()
            vy = all_targets - all_targets.mean()
            corr = (torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8)).item()
            
            metric_logger.update(mse=mse, mae=mae, r2=r2, corr=corr)

    metric_logger.synchronize_between_processes()
    
    if rank == 0:
        print(f"[{mode.upper()}] Global stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    """Main fine-tuning function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='fMRI Downstream Fine-tuning')
    parser.add_argument('--config', type=str, default='configs/finetune_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
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
            f.write(f"Fine-tuning started at {datetime.datetime.now()}\n")
            f.write("="*80 + "\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Output directory: {config['experiment']['output_dir']}\n")
            f.write(f"Task type: {config['task']['task_type']}\n")
    else:
        checkpoint_dir = None
        log_file = None

    if is_distributed:
        dist.barrier()

    # Print configuration
    if rank == 0:
        print(f"Config: {args.config}")
        print(f"Output directory: {config['experiment']['output_dir']}")
        print(f"Task type: {config['task']['task_type']}")
        print(f"Num classes: {config['task']['num_classes']}")
        print(f"Distributed: {is_distributed}")

    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(config)
    model = model.cuda(gpu)

    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    model_without_ddp = model.module if is_distributed else model

    # Optionally freeze the encoder
    if config['training'].get('freeze_encoder', True):
        if rank == 0:
            print("Freezing encoder weights. Only the head will be trained.")
        for name, param in model_without_ddp.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

        # Log which parameters are trainable
        if rank == 0:
            print("Trainable parameters:")
            for name, param in model_without_ddp.named_parameters():
                if param.requires_grad:
                    print(name)

    # Count and print model parameters before wrapping with DDP
    if rank == 0:
        print("\nAnalyzing model architecture...")
        count_parameters(model_without_ddp, verbose=True)

    # Create dataloaders
    if rank == 0:
        print("Creating dataloaders...")
    train_loader, val_loader, test_loader, train_sampler = create_dataloaders(
        config, is_distributed, rank, world_size
    )

    label_scaler = None
    if config['task']['task_type'] == 'regression':
        if rank == 0:
            mean_val = config['task']['mean']
            scale_val = config['task']['std']
            print(f"StandardScaler fit complete. Mean: {mean_val:.4f}, Std: {scale_val:.4f}")

        norm_mean = torch.tensor(mean_val, device=gpu, dtype=torch.float32)
        norm_std = torch.tensor(scale_val, device=gpu, dtype=torch.float32)

        if is_distributed:
            dist.broadcast(norm_mean, src=0)
            dist.broadcast(norm_std, src=0)

        label_scaler = LabelScaler(norm_mean, norm_std)

    if rank == 0:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batches per epoch: {len(train_loader)}")

    task_config = config['task']
    if task_config['task_type'] == 'classification':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    else:  
        criterion = nn.MSELoss()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model_without_ddp, config)
    scheduler = create_lr_scheduler(optimizer, config, len(train_loader))

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None

    # Load checkpoint if resuming
    start_epoch = 0
    best_metric = 0.0  # For classification: accuracy
    best_loss = float('inf') # For regression: loss

    if config['experiment'].get('resume', None) is not None:
        start_epoch, best_metric, best_loss = load_checkpoint(
            config['experiment']['resume'],
            model_without_ddp,
            optimizer,
            scheduler,
            scaler
        )
        print(f"Resumed from epoch {start_epoch}. Best metric: {best_metric:.4f}, Best loss: {best_loss:.4f}")
    else:
        # Initialize best_metric for new run based on task
        if config['task']['task_type'] == 'classification':
            best_metric = 0.0  # Accuracy starts at 0
        else: # regression
            best_metric = float('inf') # We use best_loss for regression, but this keeps it consistent

    # Training loop
    if rank == 0:
        print("Starting fine-tuning...")
        print(f"Training from epoch {start_epoch} to {config['training']['epochs']}")


    for epoch in range(start_epoch, config['training']['epochs']):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, config, rank, label_scaler
        )

        # Log training stats
        if rank == 0:
            log_msg = f"Epoch {epoch} Training - "
            log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in train_stats.items()])
            print(log_msg)
            log_to_file(log_file, log_msg)

        # Validate
        if epoch % config['validation']['val_freq'] == 0 or epoch == config['training']['epochs'] - 1:
            val_stats = evaluate(model, val_loader, criterion, config, rank, epoch, label_scaler, 'val')
            test_stats = evaluate(model, test_loader, criterion, config, rank, epoch, label_scaler, 'test')

            # Log validation stats
            if rank == 0:
                log_msg = f"Epoch {epoch} Validation - "
                log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in val_stats.items()])
                print(log_msg)
                log_to_file(log_file, log_msg)

                log_msg = f"Epoch {epoch} Test - "
                log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in test_stats.items()])
                print(log_msg)
                log_to_file(log_file, log_msg)

            # Determine best model based on task type
            if rank == 0:
                if task_config['task_type'] == 'classification':
                    current_metric = val_stats.get('acc', 0.0)
                    is_best = current_metric > best_metric
                    if is_best:
                        best_metric = current_metric
                        best_loss = val_stats['loss']
                else:
                    is_best = val_stats['loss'] < best_loss
                    if is_best:
                        best_loss = val_stats['loss']
                        best_metric = -best_loss  # Store negative loss as metric

                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'best_loss': best_loss,
                    'config': config,
                    'train_stats': train_stats,
                    'val_stats': val_stats,
                }

                if scaler is not None:
                    checkpoint_state['scaler_state_dict'] = scaler.state_dict()

                
                save_checkpoint(
                    checkpoint_state,
                    is_best,
                    checkpoint_dir,
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )

                checkpoint_msg = f"Checkpoint saved at epoch {epoch}"
                print(checkpoint_msg)
                log_to_file(log_file, checkpoint_msg)

                if is_best:
                    if task_config['task_type'] == 'classification':
                        best_msg = f"New best validation accuracy: {best_metric:.4f}"
                    else:
                        best_msg = f"New best validation loss: {best_loss:.4f}"
                    print(best_msg)
                    log_to_file(log_file, best_msg)

        # Save periodic checkpoint
        if rank == 0 and (epoch + 1) % config['logging']['save_freq'] == 0:
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'best_loss': best_loss,
                'config': config,
            }

            if scaler is not None:
                checkpoint_state['scaler_state_dict'] = scaler.state_dict()

            save_checkpoint(
                checkpoint_state,
                False,
                checkpoint_dir,
                filename=f'checkpoint_epoch_{epoch}.pth'
            )


    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()



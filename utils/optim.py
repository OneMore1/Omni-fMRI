from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import os

import torch
import numpy as np

def setup_lora(model, config):
    """
    Apply LoRA to the model based on configuration
    """

    lora_conf = config.get('lora', {})
    if not lora_conf.get('enabled', False):
        return model

    print("Setting up LoRA...")
    target_modules = lora_conf.get('target_modules', ["qkv", "proj", "fc1", "fc2"])
    
    peft_config = LoraConfig(
        r=lora_conf.get('r', 8),
        lora_alpha=lora_conf.get('alpha', 16),
        target_modules=target_modules,
        lora_dropout=lora_conf.get('dropout', 0.1),
        bias=lora_conf.get('bias', "none"),
        modules_to_save=["head"] 
    )

    model = get_peft_model(model, peft_config)
    
    print("LoRA Setup Complete.")
    model.print_trainable_parameters()
    
    return model


def create_optimizer(model, config):
    """
    Create optimizer with Layer-wise Learning Rate Decay (LLRD) support.
    """
    train_config = config['training']
    lora_config = config['lora']
    
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    head_lr = train_config.get('head_lr', base_lr)
    layer_decay = train_config.get('layer_decay', 1.0)
    
    if lora_config['enabled']:
        print("LoRA enabled: Using simple optimizer for trainable parameters only.")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        if train_config['optimizer'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=base_lr,
                betas=tuple(train_config['betas']),
                weight_decay=weight_decay
            )
        elif train_config['optimizer'].lower() == 'sgd':
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=base_lr,
                momentum=train_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        return optimizer

    if layer_decay >= 1.0:
        print(f"Layer decay not enabled (value: {layer_decay}). Using standard optimizer.")
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'head' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)
        
        param_groups = [
            {'params': encoder_params, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
        ]
    else:
        print(f"Applying Layer-wise Learning Rate Decay (decay rate: {layer_decay})")
        param_groups = []
        try:
            num_layers = len(model.blocks) 
        except:
            num_layers = config['model'].get('depth', 12)
            
        print(f"Detected total layers: {num_layers}")
        total_depth = num_layers + 1

        def get_layer_id(name):
            # 1. Input Embeddings (Patch Embed / Pos Embed) -> Layer 0
            if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
                return 0

            elif name.startswith("blocks") or name.startswith("layers"):
                try:
                    return int(name.split('.')[1]) + 1
                except:
                    return 0
        
            elif name.startswith("norm.") or name.startswith("fc_norm."):
                return total_depth
            
            else:
                return 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias"):
                this_weight_decay = 0.0
            else:
                this_weight_decay = weight_decay

            if 'head' in name:
                param_groups.append({
                    'params': [param],
                    'lr': head_lr, 
                    'weight_decay': this_weight_decay,
                    'name': name
                })
                continue
            layer_id = get_layer_id(name)
            
            scale = layer_decay ** (total_depth - layer_id)
            group_lr = base_lr * scale
            
            param_groups.append({
                'params': [param],
                'lr': group_lr,
                'weight_decay': this_weight_decay,
                'name': name
            })

    if train_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(train_config['betas']),
            weight_decay=weight_decay 
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=weight_decay
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
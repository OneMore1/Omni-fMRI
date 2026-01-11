"""
Regression task for brain network analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import os
from sklearn.metrics import r2_score


class RegressionTask:
    """
    Regression task handler for brain network analysis.
    
    Args:
        model (nn.Module): The model to train
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        config (dict): Configuration dictionary
    """
    
    def __init__(self, model, train_dataset, val_dataset=None, test_dataset=None, config=None, collate_fn=None, logger=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config or {}
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # saving/loading configs
        self.ckpt_dir = self.config.get('checkpoint_dir', 'checkpoints')
        self.auto_resume = self.config.get('auto_resume', True)
        self.start_epoch = 0
        self.n_iters = 0
        self.logger = logger
        self.log_every = self.config.get('log_every', 10)
        self.exp_name = self.config.get('experiment_name', 'regression_experiment')
        
        # Hyperparameters
        hparams = self.config.get('task', {}).get('params', {})
        # Hyperparameters
        self.batch_size = hparams.get('batch_size', 32)
        self.num_epochs = hparams.get('num_epochs', 100)
        self.learning_rate = hparams.get('learning_rate', 0.001)
        self.weight_decay = hparams.get('weight_decay', 5e-4)
        self.num_workers = hparams.get('num_workers', 8)
        self.collate_fn = collate_fn
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                               mode='min', 
                                                               patience=10, 
                                                               factor=0.5)
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=self.num_workers,
                                       collate_fn=self.collate_fn)
        if val_dataset:
            self.val_loader = DataLoader(val_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        num_workers=self.num_workers,
                                        collate_fn=self.collate_fn)
        if test_dataset:
            self.test_loader = DataLoader(test_dataset, 
                                         batch_size=self.batch_size, 
                                         shuffle=False, 
                                         num_workers=self.num_workers,
                                         collate_fn=self.collate_fn)
        
        # History
        self.history = {
            'train_loss': [],
            'train_mse': [],
            'val_loss': [],
            'val_mse': [],
            'train_r2': [],  # r squared
            'val_r2': []
        }

        # save initial config
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        OmegaConf.save(OmegaConf.create(self.config), f"{self.ckpt_dir}/config.yaml")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        train_labels = []
        train_preds = []
        
        for batch in tqdm(self.train_loader, desc='Training'):
            self.n_iters += 1
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs = self.model(data).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss.item()

            if self.n_iters % self.log_every == 0:
                self.logger.info(f"Batch {self.n_iters:<5}, Loss: {loss.item():.4f}")

            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(outputs.cpu().detach().numpy())

        avg_loss = total_loss / len(self.train_loader)
        avg_mse = total_mse / len(self.train_loader)
        train_r2 = r2_score(train_labels, train_preds)

        return avg_loss, avg_mse, train_r2
    
    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return 0, 0
        
        self.model.eval()
        total_loss = 0
        total_mse = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for batch in self.val_loader:
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device).float()
                
                outputs = self.model(data).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                total_mse += loss.item()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(outputs.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader)
        avg_mse = total_mse / len(self.val_loader)
        val_r2 = r2_score(val_labels, val_preds)

        return avg_loss, avg_mse, val_r2

    def test(self, load_best=True):
        """Test the model."""
        if not self.test_dataset:
            return 0
        
        # load best model
        best_model_path = os.path.join(self.ckpt_dir, 'best_model.pth' if load_best else 'last.pth')
        if os.path.exists(best_model_path):
            self.logger.info(f"Loading best model from: {best_model_path}")
            self.load_checkpoint(best_model_path)

        self.model.eval()
        total_mse = 0
        test_labels = []
        test_preds = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device).float()
                
                outputs = self.model(data).squeeze()
                loss = self.criterion(outputs, labels)
                total_mse += loss.item()

                test_labels.extend(labels.cpu().numpy())
                test_preds.extend(outputs.cpu().numpy())
        
        avg_mse = total_mse / len(self.test_loader)
        test_r2 = r2_score(test_labels, test_preds)

        self.logger.info("================== Test Results =================")
        self.logger.info(f"Test MSE: {avg_mse:.4f}, Test R2: {test_r2:.4f}")
        self.logger.info("================== Test Results =================")

        with open(f"{self.ckpt_dir}/test_results.csv", 'w') as f:
            f.write("mse,r2\n")
            f.write(f"{avg_mse},{test_r2}\n")

        return avg_mse, test_r2

    def train(self):
        """Train the model for multiple epochs."""

        if self.auto_resume:
            checkpoint_path = os.path.join(self.ckpt_dir, 'last.pth')
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)

        best_val_mse = float('inf')
        
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info("--------------------------------------------------")
            self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")

            train_loss, train_mse, train_r2 = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_mse'].append(train_mse)
            self.history['train_r2'].append(train_r2)

            self.logger.info("---------------------------")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}")

            if self.val_dataset:
                val_loss, val_mse, val_r2 = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_mse'].append(val_mse)
                self.history['val_r2'].append(val_r2)

                self.logger.info("---------------------------")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")

                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    self.save_checkpoint(f'{self.ckpt_dir}/best_model.pth')
                    
            # Save last checkpoint
            self.save_checkpoint(f'{self.ckpt_dir}/last.pth')
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': len(self.history['train_loss']),
            'history': self.history
        }, filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch']
        self.n_iters = self.start_epoch * len(self.train_loader)

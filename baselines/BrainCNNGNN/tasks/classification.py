"""
Classification task for brain network analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import os


class ClassificationTask:
    """
    Classification task handler for brain network analysis.
    
    Args:
        model (nn.Module): The model to train
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        test_dataset (Dataset): Test dataset
        config (dict): Configuration dictionary
        collate_fn (callable, optional): Optional collate function for DataLoader
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
        self.exp_name = self.config.get('experiment_name', 'classification_experiment')
        
        # Hyperparameters
        hparams = self.config.get('task', {}).get('params', {})
        self.batch_size = hparams.get('batch_size', 32)
        self.num_epochs = hparams.get('num_epochs', 100)
        self.learning_rate = hparams.get('learning_rate', 0.001)
        self.weight_decay = hparams.get('weight_decay', 5e-4)
        self.num_workers = hparams.get('num_workers', 8)
        self.collate_fn = collate_fn
        self.num_classes = self.config['model']['params'].get('num_classes', 2)
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
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
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }

        # save initial config
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        OmegaConf.save(OmegaConf.create(self.config), f"{self.ckpt_dir}/config.yaml")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        train_labels = []
        train_preds = []

        for batch in tqdm(self.train_loader, desc='Training'):
            self.n_iters += 1
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            
            if self.n_iters % self.log_every == 0:
                self.logger.info(f"Batch {self.n_iters:<5}, Loss: {loss.item():.4f}")
            
            train_labels.append(labels.cpu().numpy())
            train_preds.append(predicted.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(np.concatenate(train_labels), np.concatenate(train_preds))
        # f1 = f1_score(np.concatenate(train_labels), np.concatenate(train_preds), average='binary' if self.num_classes==2 else 'weighted')
        f1 = f1_score(np.concatenate(train_labels), np.concatenate(train_preds), average='weighted')

        return avg_loss, accuracy, f1

    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return 0, 0
        
        self.model.eval()
        total_loss = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for batch in self.val_loader:
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)

                val_labels.append(labels.cpu().numpy())
                val_preds.append(predicted.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(np.concatenate(val_labels), np.concatenate(val_preds))
        # f1 = f1_score(np.concatenate(val_labels), np.concatenate(val_preds), average='binary' if self.num_classes==2 else 'weighted')
        f1 = f1_score(np.concatenate(val_labels), np.concatenate(val_preds), average='weighted')

        return avg_loss, accuracy, f1

    def test(self, load_best=True):
        """Test the model."""
        if not self.test_dataset:
            return 0, 0
        
        # load best model
        best_model_path = os.path.join(self.ckpt_dir, 'best_model.pth' if load_best else 'last.pth')
        if os.path.exists(best_model_path):
            self.logger.info(f"Loading best model from: {best_model_path}")
            self.load_checkpoint(best_model_path)
        
        self.model.eval()
        test_labels = []
        test_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(data)
                predicted = outputs.argmax(dim=1)
                test_labels.append(labels.cpu().numpy())
                test_preds.append(predicted.cpu().numpy())

        accuracy = accuracy_score(np.concatenate(test_labels), np.concatenate(test_preds))
        # f1 = f1_score(np.concatenate(test_labels), np.concatenate(test_preds), average='binary' if self.num_classes==2 else 'weighted')
        f1 = f1_score(np.concatenate(test_labels), np.concatenate(test_preds), average='weighted')

        self.logger.info("================== Test Results =================")
        self.logger.info(f"Test Accuracy: {accuracy*100:.2f}%, Test F1: {f1:.2f}")
        self.logger.info("================== Test Results =================")

        with open(f"{self.ckpt_dir}/test_results.csv", 'w') as f:
            f.write("accuracy,f1_score\n")
            f.write(f"{accuracy},{f1}\n")

        return accuracy, f1
    
    def train(self):
        """Train the model for multiple epochs."""

        if self.auto_resume:
            import os
            checkpoint_path = os.path.join(self.ckpt_dir, 'last.pth')
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)
        
        best_val_acc = 0

        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.info("--------------------------------------------------")
            self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")

            train_loss, train_acc, train_f1 = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.logger.info("---------------------------")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Train F1: {train_f1:.2f}")

            if self.val_dataset:
                val_loss, val_acc, val_f1 = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)

                self.logger.info("---------------------------")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.2f}")

                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
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

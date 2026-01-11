"""
Classification task for brain network analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
import os


class GraphClassificationTask:
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
        self.learning_rate = hparams.get('learning_rate', 0.01)
        self.weight_decay = hparams.get('weight_decay', 5e-3)
        self.num_workers = hparams.get('num_workers', 8)
        self.collate_fn = collate_fn
        # Loss and optimizer
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = F.nll_loss
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
        #                                                        mode='min', 
        #                                                        patience=10, 
        #                                                        factor=0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)


        self.ratio = self.config['model']['params'].get('ratio', 0.5)
        self.nclass = self.config['model']['params'].get('nclass', 2)
        self.lamb0 = hparams.get('lamb0', 1) # classification loss weight
        self.lamb1 = hparams.get('lamb1', 0) # s1 unit regularization
        self.lamb2 = hparams.get('lamb2', 0) # s2 unit regularization
        self.lamb3 = hparams.get('lamb3', 0.1) # s1 entropy regularization
        self.lamb4 = hparams.get('lamb4', 0.1) # s2 entropy regularization
        self.lamb5 = hparams.get('lamb5', 0.1) # s1 consistence regularization



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

    ############################### Define Other Loss Functions ########################################
    @staticmethod
    def topk_loss(s,ratio, EPS=1e-10):
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res

    @staticmethod
    def consist_loss(s, device):
        if len(s) == 0:
            return 0
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res


    ###################### Network Training Function#####################################
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        s1_list = []
        s2_list = []
        total_loss = 0
        train_labels = []
        train_preds = []

        # for batch in tqdm(self.train_loader, desc='Training'):
        for data in tqdm(self.train_loader, desc='Training'):
            self.n_iters += 1
            # data = batch['data'].to(self.device)
            # labels = batch['label'].to(self.device)
            data = data.to(self.device)
            labels = data.y
            
            self.optimizer.zero_grad()
            # outputs = self.model(data)
            outputs, w1, w2, s1, s2 = self.model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
            s1_list.append(s1.view(-1).detach().cpu().numpy())
            s2_list.append(s2.view(-1).detach().cpu().numpy())

            # print('outputs shape:', outputs.shape, 'data.y shape:', data.y.shape)

            loss_c = self.criterion(outputs, labels)
            # loss_c = F.nll_loss(outputs, labels)
            loss_p1 = (torch.norm(w1, p=2)-1) ** 2
            loss_p2 = (torch.norm(w2, p=2)-1) ** 2
            loss_tpk1 = self.topk_loss(s1, self.ratio)
            loss_tpk2 = self.topk_loss(s2, self.ratio)
            
            loss_consist = 0
            for c in range(self.nclass):
                loss_consist += self.consist_loss(s1[data.y == c], self.device)
            loss = self.lamb0*loss_c + self.lamb1 * loss_p1 + self.lamb2 * loss_p2 \
                    + self.lamb3 * loss_tpk1 + self.lamb4 *loss_tpk2 + self.lamb5 * loss_consist


            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)

            # print('predicted:', predicted, 'labels:', labels)

            if self.n_iters % self.log_every == 0:
                self.logger.info(f"Batch {self.n_iters:<5}, Loss: {loss.item():.4f}. "
                                 f"| classification_loss: {loss_c.item():.4f} "
                                 f"| unit_loss1: {loss_p1.item():.4f} | unit_loss2: {loss_p2.item():.4f} "
                                 f"| TopK_loss1: {loss_tpk1.item():.4f} | TopK_loss2: {loss_tpk2.item():.4f} "
                                 f"| GCL_loss: {loss_consist.item():.4f}")
            
            train_labels.extend(labels.cpu())
            train_preds.extend(predicted.cpu())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(train_labels, train_preds)
        # f1 = f1_score(train_labels, train_preds, average='binary' if self.nclass==2 else 'weighted')
        f1 = f1_score(train_labels, train_preds, average='weighted')

        return avg_loss, accuracy, f1
    
    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return 0, 0
        
        self.model.eval()
        s1_list = []
        s2_list = []
        total_loss = 0
        val_labels = []
        val_preds = []
        
        with torch.no_grad():
            # for batch in self.val_loader:
            for data in self.val_loader:
                # data = batch['data'].to(self.device)
                # labels = batch['label'].to(self.device)
                data = data.to(self.device)
                labels = data.y
                
                # outputs = self.model(data)
                outputs, w1, w2, s1, s2 = self.model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                s1_list.append(s1.view(-1).detach().cpu().numpy())
                s2_list.append(s2.view(-1).detach().cpu().numpy())
                
                # loss = self.criterion(outputs, labels)
                loss_c = self.criterion(outputs, labels)
                # loss_c = F.nll_loss(outputs, labels)
                loss_p1 = (torch.norm(w1, p=2)-1) ** 2
                loss_p2 = (torch.norm(w2, p=2)-1) ** 2
                loss_tpk1 = self.topk_loss(s1, self.ratio)
                loss_tpk2 = self.topk_loss(s2, self.ratio)
                
                loss_consist = 0
                for c in range(self.nclass):
                    loss_consist += self.consist_loss(s1[data.y == c], self.device)
                loss = self.lamb0*loss_c + self.lamb1 * loss_p1 + self.lamb2 * loss_p2 \
                        + self.lamb3 * loss_tpk1 + self.lamb4 *loss_tpk2 + self.lamb5 * loss_consist

                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                val_labels.extend(labels.cpu())
                val_preds.extend(predicted.cpu())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        # f1 = f1_score(val_labels, val_preds, average='binary' if self.nclass==2 else 'weighted')
        f1 = f1_score(val_labels, val_preds, average='weighted')

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
        s1_list = []
        s2_list = []
        test_labels = []
        test_preds = []
        
        with torch.no_grad():
            # for batch in self.test_loader:
            for data in self.test_loader:
                # data = batch['data'].to(self.device)
                # labels = batch['label'].to(self.device)
                data = data.to(self.device)
                labels = data.y
                
                # outputs = self.model(data)
                outputs, w1, w2, s1, s2 = self.model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                s1_list.append(s1.view(-1).detach().cpu().numpy())
                s2_list.append(s2.view(-1).detach().cpu().numpy())

                loss_c = self.criterion(outputs, labels)
                # loss_c = F.nll_loss(outputs, labels)
                loss_p1 = (torch.norm(w1, p=2)-1) ** 2
                loss_p2 = (torch.norm(w2, p=2)-1) ** 2
                loss_tpk1 = self.topk_loss(s1, self.ratio)
                loss_tpk2 = self.topk_loss(s2, self.ratio)
                
                loss_consist = 0
                for c in range(self.nclass):
                    loss_consist += self.consist_loss(s1[data.y == c], self.device)
                loss = self.lamb0*loss_c + self.lamb1 * loss_p1 + self.lamb2 * loss_p2 \
                        + self.lamb3 * loss_tpk1 + self.lamb4 * self.lamb4 * loss_tpk2 + self.lamb5 * loss_consist

                predicted = outputs.argmax(dim=1)

                test_labels.extend(labels.cpu())
                test_preds.extend(predicted.cpu())

        accuracy = accuracy_score(test_labels, test_preds)
        # f1 = f1_score(test_labels, test_preds, average='binary' if self.nclass==2 else 'weighted')
        f1 = f1_score(test_labels, test_preds, average='weighted')

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

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Train F1: {train_f1:.2f}")

            if self.val_dataset:
                val_loss, val_acc, val_f1 = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)

                self.logger.info("---------------------------")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.2f}")

                # Learning rate scheduling
                self.scheduler.step()
                
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

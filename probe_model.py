import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm


class LinearProbe(nn.Module):
    def __init__(self, device, num_input_features, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(num_input_features, num_classes)

        # init weights using He initialization, read its good for linear probes?
        init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        init.zeros_(self.linear.bias)

        self.to(device)
    
    def forward(self, x):
        return self.linear(x)


class TrainerConfig:
    # optimization parameters
    num_epochs = 10
    batch_size = 256
    learning_rate = 0.001
    weight_decay = 1e-3 
    # checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, device, model, train_dataset, test_dataset, config: TrainerConfig):
        self.device = device
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
            
        # log for plotting
        self.train_loss_log = []
        self.test_loss_log = []
        self.train_acc_log = []
        self.test_acc_log = []


    def save_checkpoint(self):
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(self.model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))                


    def save_metrics(self):
        metrics_path = os.path.join(self.config.ckpt_path, 'metrics.json')
        metrics = {
            'train_loss': self.train_loss_log,
            'train_acc': self.train_acc_log,
            'test_loss': self.test_loss_log,
            'test_acc': self.test_acc_log
        }
        data = {
            "metrics": metrics,
            "config": self.config.__dict__,
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=4)


    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        def run_epoch(split: str):
            is_train = split == 'train'
            self.model.train(is_train)

            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=is_train)

            losses = []
            epoch_hits = 0 
            epoch_samples = 0 

            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits = self.model(x)
                    loss = criterion(logits, y)

                    losses.append(loss.item())

                    _, y_hat = torch.max(logits.data, 1)
                    hits = (y_hat == y).float()
                    epoch_hits += hits.sum().item() 
                    epoch_samples += y.size(0) 

                if is_train:
                    # Update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            mean_loss = float(np.mean(losses))
            mean_acc = epoch_hits / epoch_samples 

            # Logging accuracy/loss
            if is_train:
                self.train_loss_log.append(mean_loss)
                self.train_acc_log.append(mean_acc)
            else:
                self.test_loss_log.append(mean_loss)
                self.test_acc_log.append(mean_acc)

            return mean_loss, mean_acc

        best_loss = float('inf')
        bar = tqdm(range(self.config.num_epochs))
        bar.set_description("Epoch 0; no stats yet")
        for epo in bar:
            train_loss, train_acc = run_epoch('train')
            # save whenever we hit new best test loss
            test_loss, test_acc = run_epoch('test')
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_checkpoint()
            self.save_metrics()

            desc = f"Epoch {epo+1}; Train Loss: {train_loss:.5f}; Train Acc: {train_acc*100:.3f}%; Test Acc: {test_acc*100:.3f}%"
            bar.set_description(desc)


    def generate_report(self, split: str ='test'):
        is_train = split == 'train'

        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=False)
        
        self.model.eval() 
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                logits = self.model(x)
                _, preds = torch.max(logits, dim=1)
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        return classification_report(y_true, y_pred, digits=4)
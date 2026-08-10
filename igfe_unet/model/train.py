import os
import sys
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils.utils_process import get_filepath, write_config, construct_paths
from utils.utils_training import load_mse_data, load_eleres_data, load_totres_data, save_train
from architectures.unet import UNet
from architectures.fno import build_fno_model
from architectures.losses import LocResloss, GloResloss, LocMixloss, GloMixloss
from architectures.feminfo import LocRes, GloRes
import time

class BaseTrainer:
    def __init__(self, net, device, cfg):
        self.net = net.to(device)
        self.device = device
        self.cfg = cfg
        self.method = cfg.method
        self.patience_stop = cfg.patience_stop
        self.patience_lr = cfg.patience_lr
        self.ckpt_name = construct_paths(cfg)[0]
        self.history = {
            'train_loss': [], 'valid_loss': [],
            'train_mae': [], 'valid_mae': []
        }

    def create_lr_scheduler(self, optimizer):
        # Unified scheduler setup for fair comparison across all loss methods.
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.patience_lr,
            min_lr=1e-6,
            threshold=1e-4,
            threshold_mode='rel',
        )

    def load_checkpoint(self):
        filepath = get_filepath(self.cfg.method)
        ckpt_name, _, _ = construct_paths(self.cfg)
        write_config(self.cfg, filepath)
        if os.path.exists(ckpt_name) and self.cfg.preload:
            self.net.load_state_dict(torch.load(ckpt_name, map_location=self.device, weights_only=True))
        return ckpt_name

    def run_training(self):
        start_time = time.time()
        train_loader, val_loader = self.data_loading()
        optimizer = Adam(self.net.parameters(), lr=self.cfg.lr_start)
        best_valid_mae = float('inf')
        if self.method == 'LocResloss':
            sample = LocRes(self.cfg)
        elif self.method == 'GloResloss':
            sample = GloRes(self.cfg)
        else:
            sample = None
        lr_sched = self.create_lr_scheduler(optimizer)
        counter = 0
        for epoch in range(self.cfg.n_epochs):
            train_loss, train_mae = self.train_model(train_loader, optimizer, sample, mode='train')
            self.update_history(train_loss, train_mae, mode='train')
            valid_loss, valid_mae = self.train_model(val_loader, optimizer, sample, mode='validate')
            self.update_history(valid_loss, valid_mae, mode='validate')
            lr_sched.step(valid_mae)
            # Checkpoint saving logic
            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                torch.save(self.net.state_dict(), self.ckpt_name)
                counter = 0
                print(f'Model saved at {self.ckpt_name}')
            else:
                counter += 1
                if counter >= self.patience_stop:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            print(f'Epoch {epoch + 1}/{self.cfg.n_epochs}, Train Loss: {train_loss}, Train MAE: {train_mae},\
                       Validation Loss: {valid_loss}, Validation MAE: {valid_mae}')
        self.save_history()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Total time: {execution_time} sec')
        filepath = get_filepath(self.cfg.method)
        self.cfg.time = execution_time
        write_config(self.cfg, filepath)

    def update_history(self, loss, mae, mode='train'):
        if mode == 'train':
            self.history['train_loss'].append(loss)
            self.history['train_mae'].append(mae)
        else:
            self.history['valid_loss'].append(loss)
            self.history['valid_mae'].append(mae)

    def save_history(self):
        # Save history
        _, history_path, _ = construct_paths(self.cfg)
        save_train(self.history['train_loss'],
                   self.history['valid_loss'],
                   history_path + '_loss.txt')
        save_train(self.history['train_mae'],
                   self.history['valid_mae'],
                   history_path + '_mae.txt')

class MSETrainer(BaseTrainer):
    def __init__(self, net, device, cfg):
        super().__init__(net, device, cfg)
        self.criterion = nn.MSELoss().to(device)

    def data_loading(self):
        train_loader, valid_loader = load_mse_data(self.cfg)
        return train_loader, valid_loader

    def train_model(self, data_loader, optimizer, sample, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        losses = 0
        maes = 0
        with torch.set_grad_enabled(mode == 'train'):
            for input, target in data_loader:
                if mode == 'train':
                    optimizer.zero_grad()
                output = self.net(input).squeeze()
                loss = self.criterion(output, target)
                mae = torch.abs(output - target).mean()
                losses += loss.item()
                maes += mae.item()
                if mode == 'train':
                    loss.backward()
                    optimizer.step()

        losses = losses / len(data_loader)
        maes = maes / len(data_loader)
        return losses, maes

class LocResTrainer(BaseTrainer):
    def data_loading(self):
        train_loader, valid_loader = load_eleres_data(self.cfg)
        return train_loader, valid_loader

    def train_model(self, data_loader, optimizer, sample, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        losses = 0
        maes = 0
        with torch.set_grad_enabled(mode == 'train'):
            for input, target, dof, force_ele in data_loader:
                if mode == 'train':
                    optimizer.zero_grad()
                output = self.net(input).squeeze()
                # Calculate the loss and grad of the FEM term
                eleResloss, grad = LocResloss(output.detach(), dof, force_ele, sample)
                mae = torch.abs(output - target).mean()
                losses += eleResloss.item()
                maes += mae.item()
                if mode == 'train':
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    output.backward(grad)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
        losses = losses / len(data_loader)
        maes = maes / len(data_loader)
        return losses, maes

class GloResTrainer(BaseTrainer):
    def data_loading(self):
        train_loader, valid_loader = load_totres_data(self.cfg)
        return train_loader, valid_loader

    def train_model(self, data_loader, optimizer, sample, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        losses = 0
        maes = 0
        with torch.set_grad_enabled(mode == 'train'):
            for input, target, dof, force in data_loader:
                if mode == 'train':
                    optimizer.zero_grad()
                output = self.net(input).squeeze()
                # Calculate the loss and grad of the FEM term
                totResloss, grad = GloResloss(output.detach(), dof, force, sample)
                mae = torch.abs(output - target).mean()
                losses += totResloss.item()
                maes += mae.item()
                if mode == 'train':
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    output.backward(grad)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
        losses = losses / len(data_loader)
        maes = maes / len(data_loader)
        return losses, maes

class BaseMixedTrainer(BaseTrainer):
    def __init__(self, net, device, cfg):
        super().__init__(net, device, cfg)
        self.gamma = cfg.gamma
        self.history.update({
            'train_loss_phy': [], 'valid_loss_phy': [],
            'train_loss_mse': [], 'valid_loss_mse': [],
            'train_gamma': [], 'valid_gamma': []
        })

    def run_training(self):
        start_time = time.time()
        train_loader, val_loader = self.data_loading()
        optimizer = Adam(self.net.parameters(), lr=self.cfg.lr_start)
        best_valid_mae = float('inf')
        if self.method == 'LocMixloss':
            sample = LocRes(self.cfg)
        elif self.method == 'GloMixloss':
            sample = GloRes(self.cfg)
        else:
            sample = None
        lr_sched = self.create_lr_scheduler(optimizer)

        counter = 0
        for epoch in range(self.cfg.n_epochs):
            gamma = self.gamma
            train_loss, train_loss_mse, train_loss_phy, train_mae = self.train_model(train_loader, optimizer, sample, mode='train')
            self.update_history(train_loss, train_loss_mse, train_loss_phy, train_mae, gamma, mode='train')
            valid_loss, valid_loss_mse, valid_loss_phy, valid_mae = self.train_model(val_loader, optimizer, sample, mode='validate')
            self.update_history(valid_loss, valid_loss_mse, valid_loss_phy, valid_mae, gamma, mode='validate')
            lr_sched.step(valid_mae)
            # Checkpoint saving logic
            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                torch.save(self.net.state_dict(), self.ckpt_name)
                counter = 0
                print(f'Model saved at {self.ckpt_name}')
            else:
                counter += 1
                if counter >= self.patience_stop:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            print(f'Epoch {epoch + 1}/{self.cfg.n_epochs}, Train Loss: {train_loss}, Train MAE: {train_mae},\
                       Validation Loss: {valid_loss}, Validation MAE: {valid_mae}')
        self.save_history()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Total time: {execution_time} sec')
        filepath = get_filepath(self.cfg.method)
        self.cfg.time = execution_time
        write_config(self.cfg, filepath)

    def update_history(self, loss, loss_mse, loss_phy, mae, gamma, mode='train'):
        if mode == 'train':
            self.history['train_loss'].append(loss)
            self.history['train_loss_phy'].append(loss_phy)
            self.history['train_loss_mse'].append(loss_mse)
            self.history['train_mae'].append(mae)
            self.history['train_gamma'].append(gamma)
        else:
            self.history['valid_loss'].append(loss)
            self.history['valid_loss_phy'].append(loss_phy)
            self.history['valid_loss_mse'].append(loss_mse)
            self.history['valid_mae'].append(mae)
            self.history['valid_gamma'].append(gamma)

    def save_history(self):
        # Save history
        _, history_path, _ = construct_paths(self.cfg)
        save_train(self.history['train_loss'],
                   self.history['valid_loss'],
                   history_path + '_loss.txt')
        save_train(self.history['train_loss_phy'],
                   self.history['valid_loss_phy'],
                   history_path + '_loss_phy.txt')
        save_train(self.history['train_loss_mse'],
                   self.history['valid_loss_mse'],
                   history_path + '_loss_mse.txt')
        save_train(self.history['train_gamma'],
                   self.history['valid_gamma'],
                   history_path + '_gamma.txt')
        save_train(self.history['train_mae'],
                   self.history['valid_mae'],
                   history_path + '_mae.txt')

class LocMixTrainer(BaseMixedTrainer):
    def data_loading(self):
        train_loader, valid_loader = load_eleres_data(self.cfg)
        return train_loader, valid_loader

    def train_model(self, data_loader, optimizer, sample, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        losses = 0
        losses_mse = 0
        losses_phy = 0
        maes = 0
        with torch.set_grad_enabled(mode == 'train'):
            for input, target, dof, force_ele in data_loader:
                if mode == 'train':
                    optimizer.zero_grad()
                output = self.net(input).squeeze()
                # Calculate the loss and grad of the FEM term
                loss, loss_mse, loss_phy, grad = LocMixloss(output, target, dof, force_ele, sample, self.gamma)
                mae = torch.abs(output - target).mean()
                losses += loss.item()
                losses_mse += loss_mse.item()
                losses_phy += loss_phy.item()
                maes += mae.item()
                if mode == 'train':
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    output.backward(grad)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
        losses = losses / len(data_loader)
        losses_mse = losses_mse / len(data_loader)
        losses_phy = losses_phy / len(data_loader)
        maes = maes / len(data_loader)
        return losses, losses_mse, losses_phy, maes

class GloMixTrainer(BaseMixedTrainer):
    def data_loading(self):
        train_loader, valid_loader = load_totres_data(self.cfg)
        return train_loader, valid_loader

    def train_model(self, data_loader, optimizer, sample, mode='train'):
        if mode not in ['train', 'validate']:
            raise ValueError("Mode must be 'train' or 'validate'")
        self.net.train() if mode == 'train' else self.net.eval()
        losses = 0
        losses_mse = 0
        losses_phy = 0
        maes = 0
        with torch.set_grad_enabled(mode == 'train'):
            for input, target, dof, force in data_loader:
                if mode == 'train':
                    optimizer.zero_grad()
                output = self.net(input).squeeze()
                # Calculate the loss and grad of the FEM term
                loss, loss_mse, loss_phy, grad = GloMixloss(output, target, dof, force, sample, self.gamma)
                mae = torch.abs(output - target).mean()
                losses += loss.item()
                losses_mse += loss_mse.item()
                losses_phy += loss_phy.item()
                maes += mae.item()
                if mode == 'train':
                    grad_norm = grad.norm()
                    if grad_norm > 1.0:
                        grad = grad / grad_norm
                    output.backward(grad)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
        losses = losses / len(data_loader)
        losses_mse = losses_mse / len(data_loader)
        losses_phy = losses_phy / len(data_loader)
        maes = maes / len(data_loader)
        return losses, losses_mse, losses_phy, maes

class Training:
    def __init__(self, cfg):
        self.device = cfg.device
        self.cfg = cfg
        self.net = self.select_network()

    def select_network(self, *args):
        model_type = str(getattr(self.cfg, 'model_type', 'unet')).lower()
        if model_type == 'fno':
            backend = str(getattr(self.cfg, 'fno_backend', 'custom')).lower()
            if backend == 'custom':
                return build_fno_model(self.cfg)
            if backend == 'neuralop':
                from fno.neuraloperator import build_neuralop_fno_model
                return build_neuralop_fno_model(self.cfg)
            raise ValueError(f"Unsupported FNO backend: {backend}")

        if len(args) == 3:
            _, filters_list, kernel_size = args
        elif len(args) == 2:
            filters_list, kernel_size = args
        elif len(args) == 0:
            filters_list = self.cfg.filters_list
            kernel_size = self.cfg.kernel_size
        else:
            raise ValueError("select_network expects (filters_list, kernel_size).")

        use_batch_norm = bool(getattr(self.cfg, 'use_batch_norm', False))
        return UNet(filters_list, kernel_size, use_batch_norm=use_batch_norm)

    def select_trainer(self, method):
        if method == 'MSE':
            return MSETrainer(self.net, self.device, self.cfg)
        elif method == 'LocResloss':
            return LocResTrainer(self.net, self.device, self.cfg)
        elif method == 'GloResloss':
            return GloResTrainer(self.net, self.device, self.cfg)
        elif method == 'LocMixloss':
            return LocMixTrainer(self.net, self.device, self.cfg)
        elif method == 'GloMixloss':
            return GloMixTrainer(self.net, self.device, self.cfg)

    def run_train(self):
        cfg = self.cfg
        trainer = self.select_trainer(cfg.method)
        trainer.run_training()

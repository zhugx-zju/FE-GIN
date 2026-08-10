"""Training implementation shared by both FNO backends."""

import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler

from architectures.fno import build_fno_model
from .common import (
    count_parameters,
    count_tensor_parameters,
    output_root,
    run_id,
    set_seed,
    write_json,
)
from utils.utils_training import load_mse_data


class FNOTrainer:
    def __init__(self, cfg, output_root, model_builder=build_fno_model):
        self.cfg = cfg
        self.output_root = Path(output_root)
        self.experiment_id = run_id(cfg)
        self.model_dir = self.output_root / "models" / self.experiment_id
        self.log_dir = self.output_root / "logs" / self.experiment_id
        self.config_path = self.output_root / "configs" / f"{self.experiment_id}.json"
        self.checkpoint_path = self.model_dir / "model.pt"
        self.history_path = self.log_dir / "history.csv"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.net = model_builder(cfg).to(cfg.device)
        self.parameter_count = count_parameters(self.net)
        self.tensor_parameter_count = count_tensor_parameters(self.net)
        self.criterion = nn.MSELoss()

    def run_epoch(self, loader, optimizer, training):
        self.net.train(training)
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        with torch.set_grad_enabled(training):
            for inputs, targets in loader:
                inputs = inputs.to(self.cfg.device)
                targets = targets.to(self.cfg.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                predictions = self.net(inputs)
                loss = self.criterion(predictions, targets)
                mae = torch.abs(predictions - targets).mean()
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()
                batch_size = inputs.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_mae += float(mae.item()) * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise RuntimeError("The selected data split is empty")
        return total_loss / total_samples, total_mae / total_samples

    @staticmethod
    def save_history(path, history):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    def run_training(self):
        print(f"Device: {self.cfg.device}")
        print(f"Data path: {self.cfg.data_path}")
        print(f"Trainable real-scalar parameters: {self.parameter_count}")
        print(f"Trainable tensor numel: {self.tensor_parameter_count}")
        train_loader, valid_loader = load_mse_data(self.cfg)
        optimizer = Adam(
            self.net.parameters(),
            lr=self.cfg.lr_start,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.cfg.patience_lr,
            min_lr=1e-6,
        )
        best_mae = float("inf")
        best_epoch = 0
        stale_epochs = 0
        history = []
        start = time.perf_counter()

        for epoch in range(1, self.cfg.n_epochs + 1):
            train_loss, train_mae = self.run_epoch(train_loader, optimizer, True)
            valid_loss, valid_mae = self.run_epoch(valid_loader, optimizer, False)
            scheduler.step(valid_mae)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mae": train_mae,
                "valid_mae": valid_mae,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"Epoch {epoch}/{self.cfg.n_epochs}: "
                f"train_mae={train_mae:.6g}, valid_mae={valid_mae:.6g}"
            )
            if valid_mae < best_mae:
                best_mae = valid_mae
                best_epoch = epoch
                stale_epochs = 0
                torch.save(self.net.state_dict(), self.checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= self.cfg.patience_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        elapsed = time.perf_counter() - start
        self.save_history(self.history_path, history)
        config_values = {
            key: value for key, value in vars(self.cfg).items()
            if not key.startswith("_")
        }
        config_values.update({
            "parameter_count": self.parameter_count,
            "tensor_parameter_count": self.tensor_parameter_count,
            "best_epoch": best_epoch,
            "best_valid_mae": best_mae,
            "training_seconds": elapsed,
            "checkpoint": str(self.checkpoint_path),
        })
        write_json(self.config_path, config_values)
        print(f"Saved checkpoint: {self.checkpoint_path}")
        print(f"Saved config: {self.config_path}")
        print(f"Training seconds: {elapsed:.3f}")


def train_fno(cfg, model_builder=build_fno_model, output_root_override=None):
    """Train from a prepared config object, matching the U-Net model API."""
    set_seed(cfg.seed)
    root = output_root(output_root_override or cfg.output_dir)
    return FNOTrainer(cfg, root, model_builder=model_builder).run_training()


__all__ = ["FNOTrainer", "train_fno"]

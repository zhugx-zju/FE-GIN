import sys
import os
import csv
import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from .train import Training
from utils.utils_process import construct_paths, format_decimal_token
from utils.utils_test import load_test_data, load_validation_data, generate_noise_data

class Testing:
    """
    Enhanced testing class with support for:
    - All samples evaluation (for Table 1, 2, A1)
    - Selected samples evaluation (for visualization)
    - Mix dataset support (bil, exp, grf)
    - Multiple save formats (npz, mat, both)
    - Separate folders for all_samples and selected_samples
    """

    def __init__(self, cfg, experiment_path=None):
        self.device = cfg.device
        self.cfg = cfg
        self.num = cfg.num
        self.nodesx = cfg.nodesx
        self.nodesy = cfg.nodesy
        self.config_type = cfg.config_type
        self.dataset_type = getattr(cfg, 'dataset_type', self.config_type)
        split = str(getattr(cfg, 'eval_split', '')).strip().lower()
        self.eval_split = split if split in ['test', 'val'] else None

        # Get parameters from config
        self.save_format = cfg.save_format

        # Noise level only for mix dataset (robustness analysis)
        if self.dataset_type == 'mix':
            self.noise_level = cfg.noise_level if cfg.noise_level != 0 else None
            self.noise_levels = list(getattr(cfg, 'noise_levels', [0, 1, 3]))
            self.sample_index = int(getattr(cfg, 'sample_index', 0))
        else:
            self.noise_level = None
            self.noise_levels = [0]
            self.sample_index = 0

        # Get evaluation types for mix dataset
        if self.dataset_type == 'mix':
            self.eval_types = cfg.eval_types
            if self.eval_types == 'all':
                self.eval_types = ['mix', 'bil', 'exp', 'grf']
            elif isinstance(self.eval_types, str):
                self.eval_types = [self.eval_types]
        else:
            self.eval_types = None

        # Load network
        training_manager = Training(cfg)
        if experiment_path is None:
            ckpt_name, self.history_path, self.train_path = construct_paths(cfg)
        else:
            self.train_path = experiment_path
            self.history_path = os.path.join(experiment_path, 'history')
            ckpt_name = os.path.join(experiment_path, 'model.pt')
        self.net = training_manager.select_network()
        self.net.load_state_dict(torch.load(ckpt_name, map_location=self.device, weights_only=True))
        self.net.to(self.device)
        self.net.eval()

        # Load data
        self._load_data()

    @staticmethod
    def _relative_l1_error(target_field, pred_field):
        """Compute full-field relative L1 error over all pixels."""
        target_flat = np.asarray(target_field).reshape(-1)
        pred_flat = np.asarray(pred_field).reshape(-1)
        denom = np.sum(np.abs(target_flat))
        if np.isclose(denom, 0.0):
            return 0.0
        return float(np.sum(np.abs(target_flat - pred_flat)) / denom)

    def _load_data(self):
        """Load test data based on config type."""
        load_split_data = load_validation_data if self.eval_split == 'val' else load_test_data

        if self.dataset_type == 'mix':
            # Load data for each type
            self.data_dict = {}
            original_data_path = self.cfg.data_path
            loaded_types = []

            for data_type in self.eval_types:
                # Construct path for specific type
                eval_data_paths = getattr(self.cfg, 'eval_data_paths', {})
                if data_type in eval_data_paths:
                    base_path = eval_data_paths[data_type]
                elif data_type == 'mix':
                    base_path = original_data_path
                else:
                    base_path = original_data_path.replace('/data_mix/', f'/data_{data_type}/')
                    if base_path == original_data_path:
                        base_path = original_data_path.replace('data_mix', f'data_{data_type}')    
        
                self.cfg.data_path = base_path
                try:
                    inputs, targets = load_split_data(self.cfg)
                    self.data_dict[data_type] = (inputs, targets)
                    loaded_types.append(data_type)
                    print(f"Loaded {data_type} dataset: {inputs.shape[0]} samples")
                except FileNotFoundError as e:
                    print(f"Warning: Could not load {data_type} dataset from {base_path}")
                    print(f"Error: {e}")

            # Update to only successfully loaded types
            self.eval_types = loaded_types
            if not self.eval_types:
                raise RuntimeError("No datasets could be loaded for mix config type")

            # Restore original path
            self.cfg.data_path = original_data_path
        else:
            # Load single dataset
            inputs, targets = load_split_data(self.cfg)
            self.inputs = inputs
            self.targets = targets

    def compute_and_save_predictions(self):
        """Main evaluation function."""
        if self.dataset_type == 'mix':
            # Evaluate each type separately
            for data_type in self.eval_types:
                print(f"\nEvaluating {data_type} dataset...")
                inputs, targets = self.data_dict[data_type]
                self._evaluate_dataset(inputs, targets, data_type=data_type)
        else:
            # Evaluate single dataset
            self._evaluate_dataset(self.inputs, self.targets)

    def compute_and_save_sample_panels(self):
        """Generate Chapter 6 style 3x4 panels for a fixed sample index."""
        if self.dataset_type == 'mix':
            for data_type in self.eval_types:
                print(f"\nGenerating sample panel for {data_type} dataset...")
                inputs, targets = self.data_dict[data_type]
                self._plot_sample_panel(inputs, targets, data_type=data_type)
        else:
            self._plot_sample_panel(self.inputs, self.targets)

    def _plot_sample_panel(self, inputs, targets, data_type=None):
        device = self.device
        idx = min(max(self.sample_index, 0), inputs.shape[0] - 1)
        noise_levels = self.noise_levels if self.noise_levels else [0]

        panel_data = []
        self.net.eval()
        with torch.no_grad():
            for noise_level in noise_levels:
                input_tmp = inputs[idx].to(device)
                target_tmp = targets[idx].to(device)

                if noise_level > 0:
                    input_tmp = generate_noise_data(input_tmp, noise_level, seed=idx)

                if input_tmp.dim() == 3:
                    input_tmp = input_tmp.unsqueeze(0)
                output_tmp = self.net(input_tmp).squeeze()

                target_np = target_tmp.squeeze().cpu().numpy()
                pred_np = output_tmp.cpu().numpy()
                abs_err = np.abs(target_np - pred_np)

                l1 = self._relative_l1_error(target_np, pred_np)
                rel_err = np.zeros_like(target_np)
                mask = ~np.isclose(target_np, 0)
                rel_err[mask] = (abs_err[mask] / target_np[mask]) * 100

                panel_data.append({
                    'noise': noise_level,
                    'target': target_np,
                    'pred': pred_np,
                    'rel_err': rel_err,
                    'l1': l1
                })

        # Keep prediction colorbar consistent with true field range.
        mod_vmin = float(np.min(panel_data[0]['target']))
        mod_vmax = float(np.max(panel_data[0]['target']))
        # Use fixed upper limit for relative error map.
        err_vmax = 20.0
        # Use shared scatter limits across all noise levels for direct row-by-row comparison.
        all_true = np.concatenate([d['target'].reshape(-1) for d in panel_data])
        all_pred = np.concatenate([d['pred'].reshape(-1) for d in panel_data])
        scatter_min = min(np.min(all_true), np.min(all_pred))
        scatter_max = max(np.max(all_true), np.max(all_pred))

        n_rows = len(panel_data)
        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows), squeeze=False)
        col_titles = [
            'True Modulus Field',
            'Predicted Modulus Field',
            'Relative Error Map (%)',
            r'$\hat{E}$ versus $E^{true}$'
        ]

        for row, data in enumerate(panel_data):
            noise_label = f"Noise Level = {data['noise']:g}%"

            ax_true = axes[row, 0]
            im_true = self._plot_contour_field(ax_true, data['target'], cmap='jet', vmin=mod_vmin, vmax=mod_vmax)
            plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.02)
            ax_true.text(-0.05, 1.05, noise_label, transform=ax_true.transAxes, fontsize=12, fontweight='bold')

            ax_pred = axes[row, 1]
            im_pred = self._plot_contour_field(ax_pred, data['pred'], cmap='jet', vmin=mod_vmin, vmax=mod_vmax)
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.02)

            ax_err = axes[row, 2]
            im_err = self._plot_contour_field(ax_err, data['rel_err'], cmap='hot', vmin=0.0, vmax=err_vmax)
            plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.02)

            ax_scatter = axes[row, 3]
            x_true = data['target'].reshape(-1)
            y_pred = data['pred'].reshape(-1)
            ax_scatter.plot([scatter_min, scatter_max], [scatter_min, scatter_max],
                            color='red', linewidth=1.2, label=r'$y=x$')
            ax_scatter.scatter(
                x_true, y_pred, s=8, facecolors='none',
                edgecolors='deepskyblue', linewidths=0.5, label='Predicted Values'
            )
            ax_scatter.text(0.03, 0.97, f"$L_1$-error = {data['l1']:.2e}",
                            transform=ax_scatter.transAxes, va='top', fontsize=10)
            ax_scatter.set_xlim(scatter_min, scatter_max)
            ax_scatter.set_ylim(scatter_min, scatter_max)
            ax_scatter.set_aspect('equal', adjustable='box')
            ax_scatter.set_xlabel('True Modulus Distribution')
            ax_scatter.set_ylabel('Predicted Modulus Distribution')
            ax_scatter.legend(loc='upper left', fontsize=9, frameon=False)

            if row == 0:
                for col in range(4):
                    axes[row, col].set_title(col_titles[col], fontsize=13)

        plt.tight_layout()

        folder_name = f"selected_samples_{self.eval_split}" if self.eval_split else 'selected_samples'
        save_dir = os.path.join(self.train_path, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        type_str = f"_{data_type}" if data_type else ""
        panel_path = os.path.join(save_dir, f'sample_{idx}_panel{type_str}.png')
        plt.savefig(panel_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved sample panel to: {panel_path}")

    def _evaluate_dataset(self, inputs, targets, data_type=None):
        """
        Evaluate a dataset.

        Args:
            inputs: Input tensor
            targets: Target tensor
            data_type: 'bil', 'exp', 'grf', or None for non-mix datasets
        """
        num = self.num
        device = self.device
        nodesx = self.nodesx
        nodesy = self.nodesy

        # Determine evaluation mode
        if self.num == 'all':
            # Evaluate all samples
            num = inputs.shape[0]
            eval_inputs = inputs
            eval_targets = targets
            is_all_samples = True
        else:
            # Evaluate selected samples
            num = min(self.num, inputs.shape[0])
            eval_inputs = inputs[0:num]
            eval_targets = targets[0:num]
            is_all_samples = False

        # Create memory variables
        org = np.zeros((num, nodesy, nodesx))
        pred = np.zeros((num, nodesy, nodesx))
        error = np.zeros((num, nodesy, nodesx))
        L1 = np.zeros(num)
        MAE = np.zeros(num)
        RMSE = np.zeros(num)
        target_mean = np.zeros(num)
        target_std = np.zeros(num)
        pred_mean = np.zeros(num)
        pred_std = np.zeros(num)

        # Evaluate on GPU
        print(f"Evaluating {num} samples...")
        self.net.eval()
        with torch.no_grad():
            for i in range(num):
                # Move data to GPU
                inputs_tmp = eval_inputs[i].to(device)
                targets_tmp = eval_targets[i].to(device)

                # Apply noise using Equation 19 method
                if self.noise_level is not None and self.noise_level > 0:
                    inputs_tmp = generate_noise_data(inputs_tmp, self.noise_level, seed=i)

                if inputs_tmp.dim() == 3:
                    inputs_tmp = inputs_tmp.unsqueeze(0)
                outputs_tmp = self.net(inputs_tmp).squeeze()

                # Move to CPU
                targets_tmp = targets_tmp.squeeze().cpu().numpy()
                outputs_tmp = outputs_tmp.cpu().numpy()

                absolute_errors = np.abs(targets_tmp - outputs_tmp)

                # Calculate relative L1 error
                L1[i] = self._relative_l1_error(targets_tmp, outputs_tmp)
                MAE[i] = float(np.mean(absolute_errors))
                RMSE[i] = float(np.sqrt(np.mean((targets_tmp - outputs_tmp) ** 2)))
                target_mean[i] = float(np.mean(targets_tmp))
                target_std[i] = float(np.std(targets_tmp))
                pred_mean[i] = float(np.mean(outputs_tmp))
                pred_std[i] = float(np.std(outputs_tmp))

                # Calculate relative error (avoid division by zero)
                mask = ~np.isclose(targets_tmp, 0)
                relative_errors = np.zeros_like(targets_tmp)
                relative_errors[mask] = (absolute_errors[mask] / targets_tmp[mask]) * 100

                # Plot for selected samples

                if not is_all_samples:
                    type_str = f"_{data_type}" if data_type else ""
                    folder_name = f"selected_samples_{self.eval_split}" if self.eval_split else 'selected_samples'
                    save_dir = os.path.join(self.train_path, folder_name)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'sample_{i}{type_str}.png')
                    self.plot_prediction(outputs_tmp, targets_tmp, relative_errors, save_path)
                # Save data
                org[i] = targets_tmp
                pred[i] = outputs_tmp
                error[i] = relative_errors

        # Save results
        metrics = {
            'relative_l1': L1,
            'mae': MAE,
            'rmse': RMSE,
            'target_mean': target_mean,
            'target_std': target_std,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
        }
        self._save_results(org, pred, error, L1, metrics, is_all_samples, data_type)

    def _save_results(self, org, pred, error, L1, metrics, is_all_samples, data_type=None):
        """
        Save evaluation results to appropriate folders.

        Args:
            org: Original (ground truth) data
            pred: Predicted data
            error: Relative errors
            L1: L1 errors
            is_all_samples: True if evaluating all samples
            data_type: 'bil', 'exp', 'grf', or None
        """
        # Determine folder based on evaluation mode
        if is_all_samples:
            folder_name = f"all_samples_{self.eval_split}" if self.eval_split else 'all_samples'
        else:
            folder_name = f"selected_samples_{self.eval_split}" if self.eval_split else 'selected_samples'

        # Create folder path
        save_dir = os.path.join(self.train_path, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename components
        noise_str = f"_noise_{format_decimal_token(self.noise_level)}" if self.noise_level else ""
        type_str = f"_{data_type}" if data_type else ""

        split_tag = self.eval_split if self.eval_split else 'test'

        # Save L1 errors
        if is_all_samples:
            # All samples: all_L1_test.txt/all_L1_val.txt (+ data type/noise suffix)
            l1_filename = os.path.join(save_dir, f'all_L1_{split_tag}{type_str}{noise_str}.txt')
        else:
            # Selected samples: L1_test.txt/L1_val.txt (+ data type/noise suffix)
            l1_filename = os.path.join(save_dir, f'L1_{split_tag}{type_str}{noise_str}.txt')

        np.savetxt(l1_filename, L1)
        print(f"Saved L1 errors to: {l1_filename}")

        metrics_filename = os.path.join(
            save_dir, f'metrics_{split_tag}{type_str}{noise_str}.csv'
        )
        metric_names = list(metrics.keys())
        with open(metrics_filename, 'w', newline='') as metric_file:
            writer = csv.writer(metric_file)
            writer.writerow(['sample'] + metric_names)
            for index in range(len(L1)):
                writer.writerow([index] + [float(metrics[name][index]) for name in metric_names])
        print(f"Saved metrics to: {metrics_filename}")

        # Save detailed results (only for selected samples)
        if not is_all_samples:
            # Generate sample range string for filename
            num_samples = len(L1)
            if num_samples > 1:
                sample_range = f"samples_0-{num_samples-1}"
            else:
                sample_range = "sample_0"

            base_filename = os.path.join(save_dir, f'{sample_range}_{split_tag}{type_str}{noise_str}')

            # Save in npz format
            if self.save_format in ['npz', 'both']:
                npz_filename = f"{base_filename}.npz"
                np.savez(npz_filename, org=org, pred=pred, error=error, **metrics)
                print(f"Saved results to: {npz_filename}")

            # Save in mat format
            if self.save_format in ['mat', 'both']:
                mat_filename = f"{base_filename}.mat"
                scio.savemat(mat_filename, {'org': org, 'pred': pred, 'error': error, **metrics})
                print(f"Saved results to: {mat_filename}")

    def plot_prediction(self, outputs, targets, relative_errors, save_path=None):
        """Plot prediction results."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        mod_vmin = float(np.min(targets))
        mod_vmax = float(np.max(targets))
        err_vmax = 20.0

        # Plot true modulus map
        im_true = self._plot_contour_field(axes[0], targets, cmap='jet', vmin=mod_vmin, vmax=mod_vmax)
        fig.colorbar(im_true, ax=axes[0])
        axes[0].set_title('True Modulus Map', fontsize=14)

        # Plot predicted modulus map
        im_pred = self._plot_contour_field(axes[1], outputs, cmap='jet', vmin=mod_vmin, vmax=mod_vmax)
        fig.colorbar(im_pred, ax=axes[1])
        axes[1].set_title('Predicted Modulus Map', fontsize=14)

        # Plot relative error map
        im_err = self._plot_contour_field(axes[2], np.abs(relative_errors), cmap='hot', vmin=0.0, vmax=err_vmax)
        cbar = fig.colorbar(im_err, ax=axes[2])
        cbar.set_label('Relative Error (%)')
        axes[2].set_title('Relative Error Map (%)', fontsize=14)

        # Plot comparison figure
        axes[3].plot(targets.reshape(-1, 1), targets.reshape(-1, 1),
                     label='y = x', linewidth=2, c='red')
        axes[3].scatter(targets.reshape(-1, 1), outputs.reshape(-1, 1), s=5)
        axes[3].legend(frameon=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600)
        plt.close()

    def _plot_contour_field(self, ax, field, cmap, vmin, vmax, levels=100):
        """Render field using smooth contour style, aligned with asm_unet_compare plotting."""
        field = np.asarray(field)
        ny, nx = field.shape
        x = np.arange(nx)
        y = np.arange(ny)
        xx, yy = np.meshgrid(x, y)

        cmap_obj = plt.get_cmap(cmap).copy()
        # Ensure out-of-range values are rendered with deep boundary colors instead of blank regions.
        cmap_obj.set_under(cmap_obj(0.0))
        cmap_obj.set_over(cmap_obj(1.0))

        if np.isclose(vmin, vmax):
            im = ax.imshow(field, cmap=cmap_obj, vmin=vmin - 1e-12, vmax=vmax + 1e-12)
            ax.set_xticks([])
            ax.set_yticks([])
            return im

        lvl = np.linspace(vmin, vmax, levels)
        im = ax.contourf(
            xx, yy, field,
            levels=lvl,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            extend='both',
        )
        ax.set_aspect('equal')
        ax.axis('off')
        return im

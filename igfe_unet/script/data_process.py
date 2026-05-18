import numpy as np
import sys
import torch
import scipy.io as sio
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from utils.utils_process import Config, config_from_json
from utils.utils_training import load_data
from architectures.feminfo import MeshInfo

config_type = config_from_json()

def process_single_case(case_type, device='cuda'):
    """Process a single case to generate dof.npy and force_ele.npy"""
    print(f"\n{'='*50}")
    print(f"Processing {case_type} dataset")
    print(f"{'='*50}")

    config = Config(case_type)
    data_path = config.data_path
    FEMInfo = MeshInfo(config)

    # Load Dataset
    input_name = 'input'
    output_name = 'output'
    input_data = load_data(input_name, config)
    output_data = load_data(output_name, config)

    print(f"Loaded data shapes:")
    print(f"  Input: {input_data.shape}")
    print(f"  Output: {output_data.shape}")

    # Create dof data
    Batch_num = input_data.shape[0]
    dof = torch.zeros(Batch_num, FEMInfo.nDOF, dtype=torch.float32, device=device)
    dof[:,0::2] = input_data[:,0,:,:].reshape((Batch_num,-1))
    dof[:,1::2] = input_data[:,1,:,:].reshape((Batch_num,-1))

    # Create element force data
    gaussE = FEMInfo.get_gauss_modulus(output_data)
    totK = FEMInfo.K0.reshape((4,-1,64))
    Kele = torch.einsum('ijk,kjl->ijl', gaussE, totK).reshape((Batch_num,-1,8,8))
    disp_ele = FEMInfo.get_ele_dof(dof)
    force_ele = Kele @ disp_ele
    # Assemble global force vector from element forces
    cMat_flat = torch.from_numpy((FEMInfo.cMat - 1).reshape(-1)).long().to(device)
    force = torch.zeros(Batch_num, FEMInfo.nDOF, device=device)
    force.scatter_add_(1, cMat_flat.unsqueeze(0).expand(Batch_num, -1),
                       force_ele.squeeze(3).reshape(Batch_num, -1))

    # Save to case directory
    np.save(data_path + '/dof', dof.cpu().numpy())
    np.save(data_path + '/force_ele', force_ele.cpu().numpy())
    np.save(data_path + '/force', force.cpu().numpy())

    print(f"Saved to {data_path}:")
    print(f"  dof.npy: {dof.shape}")
    print(f"  force_ele.npy: {force_ele.shape}")
    print(f"  force.npy: {force.shape}")

    return input_data, output_data, dof.cpu().numpy(), force_ele.cpu().numpy()

def create_mix_dataset(cases=['exp', 'bil', 'grf'],
                       seed=42,
                       bil_ratio=None,
                       exp_ratio=None,
                       grf_ratio=None,
                       train_rto=None,
                       valid_rto=None,
                       load_type=None,
                       verbose=True):
    """Combine processed data from multiple cases into mix dataset with specified ratios."""
    if verbose:
        print(f"\n{'='*50}")
        print(f"Creating Mixed Dataset")
        print(f"{'='*50}")

    mix_config = Config('mix')
    exp_ratio = mix_config.exp_ratio if exp_ratio is None else exp_ratio
    bil_ratio = mix_config.bil_ratio if bil_ratio is None else bil_ratio
    grf_ratio = mix_config.grf_ratio if grf_ratio is None else grf_ratio
    train_rto = mix_config.train_rto if train_rto is None else train_rto
    valid_rto = mix_config.valid_rto if valid_rto is None else valid_rto
    load_type = mix_config.load_type if load_type is None else load_type

    ratios = {'exp': exp_ratio, 'bil': bil_ratio, 'grf': grf_ratio}
    test_ratios = {'exp': 0.33, 'bil': 0.33, 'grf': 0.34}

    def compute_ratio_counts(total_count, ratio_map, case_order):
        raw_counts = np.array([total_count * float(ratio_map[c]) for c in case_order], dtype=float)
        base_counts = np.floor(raw_counts).astype(int)
        remainder = int(total_count - int(base_counts.sum()))
        if remainder > 0:
            frac = raw_counts - base_counts.astype(float)
            # Distribute the remainder to the largest fractional parts
            order = np.argsort(-frac)
            for i in range(remainder):
                base_counts[order[i % len(case_order)]] += 1
        return {c: int(base_counts[i]) for i, c in enumerate(case_order)}

    base_data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))

    # First, determine total_samples from the first case
    first_case = cases[0]
    first_case_dir = os.path.join(base_data_dir, f'data_{first_case}', load_type)
    first_input = sio.loadmat(os.path.join(first_case_dir, 'input.mat'))['U']
    total_samples = first_input.shape[0]

    total_train = int(total_samples * train_rto)
    total_val = int(total_samples * valid_rto)
    total_test = total_samples - total_train - total_val

    split_case_counts = {
        'train': compute_ratio_counts(total_train, ratios, cases),
        'val': compute_ratio_counts(total_val, ratios, cases),
        'test': compute_ratio_counts(total_test, test_ratios, cases),
    }

    sample_counts = {
        case: split_case_counts['train'][case] + split_case_counts['val'][case] + split_case_counts['test'][case]
        for case in cases
    }

    if verbose:
        print(f"\nMix configuration:")
        print(f"  Total samples (from data): {total_samples}")
        print(f"  Split ratios: train={train_rto:.2f}, val={valid_rto:.2f}, test={1.0-train_rto-valid_rto:.2f}")
        for case in cases:
            n_samples = sample_counts[case]
            print(f"  {case}: {ratios[case]:.2%} ({n_samples} samples)")
        print(f"  Split totals: train={total_train}, val={total_val}, test={total_test}")

    split_data = {
        'train': {'input': [], 'output': [], 'dof': [], 'force_ele': [], 'force': []},
        'val': {'input': [], 'output': [], 'dof': [], 'force_ele': [], 'force': []},
        'test': {'input': [], 'output': [], 'dof': [], 'force_ele': [], 'force': []},
    }
    split_counts = {k: {case: 0 for case in cases} for k in split_data.keys()}

    # Load and sample data from each case
    np.random.seed(int(seed))  # Set seed once for reproducibility

    for case in cases:
        case_dir = os.path.join(base_data_dir, f'data_{case}', load_type)
        if verbose:
            print(f"\nLoading {case} data from: {case_dir}")

        # Load .mat files
        input_data = sio.loadmat(os.path.join(case_dir, 'input.mat'))['U']
        output_data = sio.loadmat(os.path.join(case_dir, 'output.mat'))['E']

        # Load .npy files
        dof_data = np.load(os.path.join(case_dir, 'dof.npy'))
        force_ele_data = np.load(os.path.join(case_dir, 'force_ele.npy'))
        force_data = np.load(os.path.join(case_dir, 'force.npy'))

        available_samples = input_data.shape[0]
        if verbose:
            print(f"  Available samples: {available_samples}")
            print(f"  Requested total: {sample_counts[case]}")

        if available_samples != total_samples and verbose:
            print(f"  Warning: Expected {total_samples} samples but found {available_samples}")

        case_train_end = int(available_samples * train_rto)
        case_val_end = case_train_end + int(available_samples * valid_rto)

        # IMPORTANT: no leakage from *_test into mix_train/mix_val.
        # Use disjoint source pools by original index ranges.
        source_pools = {
            'train': np.arange(0, case_train_end),
            'val': np.arange(case_train_end, case_val_end),
            'test': np.arange(case_val_end, available_samples),
        }

        req_counts = {
            'train': int(split_case_counts['train'][case]),
            'val': int(split_case_counts['val'][case]),
            'test': int(split_case_counts['test'][case]),
        }

        sampled_by_split = {}
        for split_name in ['train', 'val', 'test']:
            pool_idx = source_pools[split_name]
            req = req_counts[split_name]
            if req > len(pool_idx):
                if verbose:
                    print(
                        f"  Warning: {case}-{split_name} requested {req} but pool has {len(pool_idx)}. "
                        f"Using {len(pool_idx)}."
                    )
                req = len(pool_idx)
            sampled_idx = np.random.choice(pool_idx, req, replace=False) if req > 0 else np.array([], dtype=int)
            sampled_by_split[split_name] = sampled_idx

        # Hard check: no overlap between split indices within the same case.
        train_set = set(sampled_by_split['train'].tolist())
        val_set = set(sampled_by_split['val'].tolist())
        test_set = set(sampled_by_split['test'].tolist())
        if (train_set & val_set) or (train_set & test_set) or (val_set & test_set):
            raise RuntimeError(f"Data leakage detected in case '{case}': split indices overlap.")

        for split_name in ['train', 'val', 'test']:
            idx = sampled_by_split[split_name]
            split_data[split_name]['input'].append(input_data[idx])
            split_data[split_name]['output'].append(output_data[idx])
            split_data[split_name]['dof'].append(dof_data[idx])
            split_data[split_name]['force_ele'].append(force_ele_data[idx])
            split_data[split_name]['force'].append(force_data[idx])
            split_counts[split_name][case] = int(len(idx))

        if verbose:
            print(
                f"  Sampled train/val/test = "
                f"{split_counts['train'][case]}/{split_counts['val'][case]}/{split_counts['test'][case]}"
            )

    # Merge per-split data and shuffle inside each split (same permutation for all fields)
    def merge_split_with_shared_perm(split_name):
        merged = {}
        for key in ['input', 'output', 'dof', 'force_ele', 'force']:
            merged[key] = np.concatenate(split_data[split_name][key], axis=0)
        perm = np.random.permutation(merged['input'].shape[0])
        for key in merged.keys():
            merged[key] = merged[key][perm]
        return merged

    train_merged = merge_split_with_shared_perm('train')
    val_merged = merge_split_with_shared_perm('val')
    test_merged = merge_split_with_shared_perm('test')

    train_input, val_input, test_input = train_merged['input'], val_merged['input'], test_merged['input']
    train_output, val_output, test_output = train_merged['output'], val_merged['output'], test_merged['output']
    train_dof, val_dof, test_dof = train_merged['dof'], val_merged['dof'], test_merged['dof']
    train_force_ele, val_force_ele, test_force_ele = train_merged['force_ele'], val_merged['force_ele'], test_merged['force_ele']
    train_force, val_force, test_force = train_merged['force'], val_merged['force'], test_merged['force']

    # Concatenate in train/val/test order so PartSet slicing remains consistent
    combined_input = np.concatenate([train_input, val_input, test_input], axis=0)
    combined_output = np.concatenate([train_output, val_output, test_output], axis=0)
    combined_dof = np.concatenate([train_dof, val_dof, test_dof], axis=0)
    combined_force_ele = np.concatenate([train_force_ele, val_force_ele, test_force_ele], axis=0)
    combined_force = np.concatenate([train_force, val_force, test_force], axis=0)

    if verbose:
        print(f"\nCombined shapes:")
        print(f"  Input: {combined_input.shape}")
        print(f"  Output: {combined_output.shape}")
        print(f"  DOF: {combined_dof.shape}")
        print(f"  Force_ele: {combined_force_ele.shape}")

    # Save to mix directory
    mix_dir = os.path.join(base_data_dir, 'data_mix', load_type)
    os.makedirs(mix_dir, exist_ok=True)

    sio.savemat(os.path.join(mix_dir, 'input.mat'), {'U': combined_input})
    sio.savemat(os.path.join(mix_dir, 'output.mat'), {'E': combined_output})
    np.save(os.path.join(mix_dir, 'dof.npy'), combined_dof)
    np.save(os.path.join(mix_dir, 'force_ele.npy'), combined_force_ele)
    np.save(os.path.join(mix_dir, 'force.npy'), combined_force)

    if verbose:
        print(f"\nSaved to {mix_dir}")
        print(f"Total samples: {combined_input.shape[0]}")
        for case in cases:
            n_case = sample_counts[case]
            print(f"  - {case}: {n_case} ({n_case/combined_input.shape[0]:.1%})")

        print("\nSplit composition by case:")
        for split_name in ['train', 'val', 'test']:
            total_split = sum(split_counts[split_name].values())
            detail = ", ".join([f"{case}:{split_counts[split_name][case]}" for case in cases])
            print(f"  {split_name} ({total_split}): {detail}")

    return {
        'mix_dir': mix_dir,
        'total_samples': int(combined_input.shape[0]),
        'split_sizes': {
            'train': int(train_input.shape[0]),
            'val': int(val_input.shape[0]),
            'test': int(test_input.shape[0]),
        },
        'case_split_counts': split_counts,
    }

# Main execution
if config_type in ['bil', 'exp', 'layer', 'grf']:
    # Process single case
    device = Config(config_type).device
    process_single_case(config_type, device)

elif config_type == 'mix':
    # Process all cases and create mix dataset
    device = Config('exp').device  # Use exp config for device setting

    # Process each case
    cases = ['exp', 'bil', 'grf']
    for case in cases:
        process_single_case(case, device)

    # Create mix dataset
    create_mix_dataset(cases)

print(f"\n{'='*50}")
print("Data processing complete!")
print(f"{'='*50}")

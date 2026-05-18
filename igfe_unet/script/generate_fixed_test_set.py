import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.utils_process import Config

# ============================================================================
# CUSTOMIZABLE PARAMETERS
# ============================================================================
TARGET_LOAD_TYPE = 'force_load'
TARGET_CASES = ['exp', 'bil', 'grf']
FIXED_TEST_SEED = 42
MIX_TEST_RATIOS = {'exp': 0.33, 'bil': 0.33, 'grf': 0.34}
# ============================================================================


def _load_array(data_dir, filename):
    path_mat = data_dir / f'{filename}.mat'
    path_npy = data_dir / f'{filename}.npy'

    if path_mat.exists():
        data = sio.loadmat(path_mat)
        if filename == 'input':
            return data['U']
        if filename == 'output':
            return data['E']
        raise KeyError(f'Unsupported mat filename: {filename}')

    if path_npy.exists():
        return np.load(path_npy)

    raise FileNotFoundError(
        f'Data file not found for {filename}. '
        f'Searched: {path_mat} and {path_npy}'
    )


def _save_input_output(dataset_dir, input_data, output_data):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(dataset_dir / 'input.mat', {'U': input_data})
    sio.savemat(dataset_dir / 'output.mat', {'E': output_data})


def _compute_ratio_counts(total_count, ratio_map, case_order):
    raw_counts = np.array([total_count * float(ratio_map[c]) for c in case_order], dtype=float)
    base_counts = np.floor(raw_counts).astype(int)
    remainder = int(total_count - int(base_counts.sum()))
    if remainder > 0:
        frac = raw_counts - base_counts.astype(float)
        order = np.argsort(-frac)
        for i in range(remainder):
            base_counts[order[i % len(case_order)]] += 1
    return {case: int(base_counts[i]) for i, case in enumerate(case_order)}


print('=' * 80)
print('Generate Fixed Test Set')
print('=' * 80)
print(f'Load type: {TARGET_LOAD_TYPE}')
print(f'Cases: {TARGET_CASES}')
print(f'Seed: {FIXED_TEST_SEED}')
print(f'Mix test ratios: {MIX_TEST_RATIOS}')

cfg = Config('mix')
data_root = Path(cfg.data_path).resolve().parent.parent
project_root = data_root.parent
output_root = data_root / 'fixed_test_sets' / TARGET_LOAD_TYPE
output_root.mkdir(parents=True, exist_ok=True)

print(f'Data root: {data_root}')
print(f'Output root: {output_root}')

train_rto = float(cfg.train_rto)
valid_rto = float(cfg.valid_rto)
rng = np.random.default_rng(FIXED_TEST_SEED)

case_test_data = {}
manifest = {
    'load_type': TARGET_LOAD_TYPE,
    'seed': FIXED_TEST_SEED,
    'train_rto': train_rto,
    'valid_rto': valid_rto,
    'mix_test_ratios': MIX_TEST_RATIOS,
    'case_order': TARGET_CASES,
    'datasets': {},
}

for case in TARGET_CASES:
    case_dir = data_root / f'data_{case}' / TARGET_LOAD_TYPE
    source_dir_rel = case_dir.relative_to(project_root).as_posix()
    input_data = _load_array(case_dir, 'input')
    output_data = _load_array(case_dir, 'output')

    total_samples = int(input_data.shape[0])
    if output_data.shape[0] != total_samples:
        raise ValueError(f'Input/output sample count mismatch for {case_dir}')

    train_end = int(total_samples * train_rto)
    valid_end = train_end + int(total_samples * valid_rto)
    test_indices = np.arange(valid_end, total_samples, dtype=int)

    case_test_input = input_data[test_indices]
    case_test_output = output_data[test_indices]
    _save_input_output(output_root / f'data_{case}', case_test_input, case_test_output)

    case_test_data[case] = {
        'source_dir': source_dir_rel,
        'input': case_test_input,
        'output': case_test_output,
        'source_indices': test_indices,
        'test_count': int(len(test_indices)),
    }
    manifest['datasets'][f'data_{case}'] = {
        'source_dir': source_dir_rel,
        'total_samples': total_samples,
        'test_start': int(valid_end),
        'test_end': int(total_samples - 1),
        'test_count': int(len(test_indices)),
    }

    print(f'Saved fixed test set for data_{case}: {len(test_indices)} samples')

reference_case = TARGET_CASES[0]
mix_test_total = int(case_test_data[reference_case]['test_count'])
mix_case_counts = _compute_ratio_counts(mix_test_total, MIX_TEST_RATIOS, TARGET_CASES)

mix_input_parts = []
mix_output_parts = []
mix_source_indices = {}
for case in TARGET_CASES:
    req = int(mix_case_counts[case])
    case_inputs = case_test_data[case]['input']
    case_outputs = case_test_data[case]['output']
    case_source_indices = case_test_data[case]['source_indices']

    if req > case_inputs.shape[0]:
        raise ValueError(
            f'Not enough test samples in {case} to build fixed mix test set: '
            f'requested={req}, available={case_inputs.shape[0]}'
        )

    selected_local_idx = rng.choice(case_inputs.shape[0], size=req, replace=False)
    mix_input_parts.append(case_inputs[selected_local_idx])
    mix_output_parts.append(case_outputs[selected_local_idx])
    mix_source_indices[case] = case_source_indices[selected_local_idx].astype(int).tolist()

combined_input = np.concatenate(mix_input_parts, axis=0)
combined_output = np.concatenate(mix_output_parts, axis=0)
mix_perm = rng.permutation(combined_input.shape[0])
combined_input = combined_input[mix_perm]
combined_output = combined_output[mix_perm]

_save_input_output(output_root / 'data_mix', combined_input, combined_output)
manifest['datasets']['data_mix'] = {
    'source_cases': TARGET_CASES,
    'test_count': int(combined_input.shape[0]),
    'per_case_counts': mix_case_counts,
    'selected_source_indices': mix_source_indices,
}

manifest_path = output_root / 'manifest.json'
with open(manifest_path, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print(f'Saved fixed test set for data_mix: {combined_input.shape[0]} samples')
print(f'Saved manifest: {manifest_path}')
print('=' * 80)
print('Fixed test set generation complete.')
print('=' * 80)

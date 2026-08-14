"""Run final-model inference and coordinate all GRF comparison outputs."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import (
    CONDITION_DISPLAY,
    GRF_CONDITION_ORDER,
    METHOD_ORDER,
    build_metric_row,
    load_condition,
    load_dataset_manifest,
    load_final_model,
    predict_condition,
    resolve_device,
    sha256_file,
)
from .visualization import (
    plot_case_comparison,
    plot_correlation_length_curves,
    plot_paper_table,
    save_paper_table_csv,
    summarize_metrics,
    write_csv,
)


PER_SAMPLE_FIELDS = [
    'model',
    'condition_id',
    'field_type',
    'distribution_status',
    'correlation_length_mm',
    'transition_width_10_90_mm',
    'noise_level_percent',
    'sample_id',
    'relative_l1',
    'mae',
    'rmse',
]
SUMMARY_FIELDS = [
    'model',
    'condition_id',
    'field_type',
    'distribution_status',
    'correlation_length_mm',
    'transition_width_10_90_mm',
    'noise_level_percent',
    'sample_count',
    'relative_l1_mean',
    'relative_l1_std',
    'relative_l1_ratio_to_l25',
    'mae_mean',
    'mae_std',
    'rmse_mean',
    'rmse_std',
]


def _validate_case_group(raw_cases, condition_ids, required, output_group):
    cases = []
    for raw_case in raw_cases:
        condition_id = str(raw_case['condition'])
        if condition_id not in condition_ids:
            raise KeyError(f'Configured case condition is absent from manifest: {condition_id}')
        sample_index = int(raw_case.get('sample_index', 0))
        cases.append(
            {
                'condition': condition_id,
                'sample_index': sample_index,
                'output_group': output_group,
            }
        )
    configured = {case['condition'] for case in cases}
    missing = sorted(set(required) - configured)
    if missing:
        raise ValueError(f'Missing required additional GRF cases: {missing}')
    return cases


def _validate_cases(cfg, conditions):
    condition_ids = {condition['condition_id'] for condition in conditions}
    cases = _validate_case_group(
        cfg.get('cases', []),
        condition_ids,
        {'grf_l20', 'grf_l15', 'grf_l10', 'grf_l8'},
        'cases',
    )
    supplementary_cases = _validate_case_group(
        cfg.get('supplementary_cases', []),
        condition_ids,
        {'grf_l5'},
        str(Path('supplementary') / 'cases'),
    )
    return cases, supplementary_cases


def _build_case_panels(cases, condition_data, prediction_cache, noise_levels):
    panels = {}
    for case in cases:
        condition_id = case['condition']
        _, targets = condition_data[condition_id]
        sample_index = min(max(int(case['sample_index']), 0), len(targets) - 1)
        resolved_case = {
            'condition': condition_id,
            'sample_index': sample_index,
            'output_group': case.get('output_group', 'cases'),
        }
        method_panels = {}
        for model in METHOD_ORDER:
            noise_panels = {}
            for noise_level in noise_levels:
                prediction = prediction_cache[(model, condition_id, float(noise_level))][
                    sample_index
                ]
                target = targets[sample_index]
                absolute_error = np.abs(prediction - target)
                relative_error_percent = np.zeros_like(absolute_error, dtype=float)
                nonzero = ~np.isclose(target, 0.0)
                relative_error_percent[nonzero] = (
                    100.0 * absolute_error[nonzero] / np.abs(target[nonzero])
                )
                denominator = np.sum(np.abs(target))
                relative_l1 = (
                    float(np.sum(absolute_error) / denominator)
                    if not np.isclose(denominator, 0.0)
                    else np.nan
                )
                noise_panels[float(noise_level)] = {
                    'prediction': prediction,
                    'absolute_error': absolute_error,
                    'relative_error_percent': relative_error_percent,
                    'relative_l1': relative_l1,
                }
            method_panels[model] = noise_panels
        panels[condition_id] = (resolved_case, targets[sample_index], method_panels)
    return panels


def run_generalization_comparison(cfg):
    """Evaluate all conditions and save case figures plus manuscript tables."""
    data_root = Path(cfg['data_dir']).resolve()
    output_root = Path(cfg['output_dir']).resolve()
    model_root = Path(cfg['model_root']).resolve()
    noise_levels = [float(value) for value in cfg['noise_levels']]
    device = resolve_device(cfg.get('device', 'cpu'))

    manifest_path, manifest = load_dataset_manifest(data_root)
    conditions = list(manifest['conditions'])
    cases, supplementary_cases = _validate_cases(cfg, conditions)
    condition_by_id = {condition['condition_id']: condition for condition in conditions}
    condition_data = {
        condition['condition_id']: load_condition(data_root, condition)
        for condition in conditions
    }

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / 'metrics'
    figures_dir = output_root / 'figures'
    manifests_dir = output_root / 'manifests'
    for directory in (metrics_dir, figures_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifests_dir / 'dataset_manifest.json')

    rows = []
    prediction_cache = {}
    model_manifest = {}
    labels_to_directories = {label: folder for folder, label in cfg['models'].items()}
    for model_label in METHOD_ORDER:
        if model_label not in labels_to_directories:
            continue
        model_directory = model_root / labels_to_directories[model_label]
        network, model_info = load_final_model(model_directory, device)
        model_manifest[model_label] = model_info
        print(f'[Generalization] Loaded {model_label}: {model_directory.name}')
        for condition in conditions:
            condition_id = condition['condition_id']
            inputs, targets = condition_data[condition_id]
            for noise_level in noise_levels:
                predictions = predict_condition(
                    network=network,
                    device=device,
                    inputs=inputs,
                    noise_level=noise_level,
                    batch_size=cfg['batch_size'],
                    noise_seed=int(cfg['seed']) * 10000,
                )
                prediction_cache[(model_label, condition_id, noise_level)] = predictions
                for sample_id, (target, prediction) in enumerate(zip(targets, predictions)):
                    rows.append(
                        build_metric_row(
                            model_label,
                            condition,
                            noise_level,
                            sample_id,
                            target,
                            prediction,
                        )
                    )
            print(
                f"[Generalization] model={model_label}, condition={condition_id}, "
                f"samples={len(inputs)}, noise_levels={noise_levels}"
            )

    per_sample_csv = write_csv(metrics_dir / 'per_sample_all.csv', PER_SAMPLE_FIELDS, rows)
    summaries = summarize_metrics(rows)
    summary_csv = write_csv(metrics_dir / 'ood_summary.csv', SUMMARY_FIELDS, summaries)
    stress_condition_ids = {
        case['condition'] for case in supplementary_cases
    }
    stress_summaries = [
        row for row in summaries if row['condition_id'] in stress_condition_ids
    ]
    stress_summary_csv = write_csv(
        output_root / 'supplementary' / 'metrics' / 'grf_l5_stress_summary.csv',
        SUMMARY_FIELDS,
        stress_summaries,
    )
    paper_table_csv = save_paper_table_csv(
        metrics_dir / 'grf_noise_statistics_table.csv', summaries
    )
    paper_table_png = figures_dir / 'grf_noise_statistics_table.png'
    paper_table_pdf = figures_dir / 'grf_noise_statistics_table.pdf'
    plot_paper_table(
        paper_table_png,
        paper_table_pdf,
        summaries,
        dpi=int(cfg.get('dpi', 600)),
    )
    plot_correlation_length_curves(
        output_root,
        summaries,
        dpi=int(cfg.get('dpi', 600)),
    )

    case_outputs = {}
    panels = _build_case_panels(
        cases + supplementary_cases,
        condition_data,
        prediction_cache,
        noise_levels,
    )
    for condition_id, (case, target, method_panels) in panels.items():
        prediction_path, error_path = plot_case_comparison(
            output_root,
            case,
            target,
            method_panels,
            noise_levels,
            dpi=int(cfg.get('dpi', 600)),
        )
        case_outputs[condition_id] = {
            'display_name': CONDITION_DISPLAY.get(condition_id, condition_id),
            'prediction_figure': str(prediction_path),
            'error_figure': str(error_path),
        }

    evaluation_manifest = {
        'schema_version': 2,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_manifest': str(manifest_path),
        'dataset_manifest_sha256': sha256_file(manifest_path),
        'device': str(device),
        'noise_levels_percent': noise_levels,
        'models': model_manifest,
        'cases': cases,
        'supplementary_cases': supplementary_cases,
        'case_outputs': case_outputs,
        'paper_table_conditions': list(GRF_CONDITION_ORDER),
    }
    evaluation_manifest_path = manifests_dir / 'evaluation_manifest.json'
    with evaluation_manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(evaluation_manifest, handle, indent=2)

    return {
        'per_sample_csv': per_sample_csv,
        'summary_csv': summary_csv,
        'stress_summary_csv': stress_summary_csv,
        'paper_table_csv': paper_table_csv,
        'paper_table_png': paper_table_png,
        'paper_table_pdf': paper_table_pdf,
        'evaluation_manifest': evaluation_manifest_path,
        'case_outputs': case_outputs,
    }

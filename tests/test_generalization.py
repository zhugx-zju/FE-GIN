import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "igfe_unet"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from asm_log.fgm_asm.mesh import MeshInfo
from grf_generalization.pipeline.generators import (
    build_generalization_dataset,
    generate_grf_fields,
    generate_steep_gradient_fields,
)
from grf_generalization.pipeline.metrics import apply_relative_noise, field_metrics
from grf_generalization.config import get_config
from grf_generalization.pipeline.evaluation import _build_case_panels, _validate_cases
from grf_generalization.pipeline import visualization
from grf_generalization.pipeline.sample_preview import (
    _draw_true_field,
    resolve_preview_indices,
)
from grf_generalization.pipeline.visualization import build_paper_table, summarize_metrics


class GeneralizationTests(unittest.TestCase):
    def test_plot_style_matches_existing_project_figures(self):
        self.assertEqual(matplotlib.rcParams['font.family'], ['serif'])
        self.assertEqual(matplotlib.rcParams['font.serif'][0], 'Times New Roman')
        self.assertEqual(matplotlib.rcParams['mathtext.rm'], 'Times New Roman')
        self.assertEqual(matplotlib.rcParams['font.size'], 10.0)
        self.assertEqual(matplotlib.rcParams['axes.linewidth'], 0.8)
        self.assertEqual(visualization._row_tag(0), '(a)')
        self.assertEqual(visualization._row_tag(5), '(f)')

        figure, axis = plt.subplots()
        image = axis.imshow(np.arange(4).reshape(2, 2))
        colorbar = figure.colorbar(image)
        visualization._style_field_colorbar(colorbar, 'Modulus (MPa)')
        self.assertEqual(colorbar.ax.yaxis.label.get_fontweight(), 'normal')
        self.assertEqual(colorbar.ax.yaxis.label.get_fontsize(), 15.0)
        self.assertEqual(colorbar.ax.yaxis.get_major_formatter().format_data(1.25), '1.25')
        plt.close(figure)

    def test_case_panels_include_project_relative_error_percent(self):
        target = np.asarray([[1.0, 2.0], [4.0, 8.0]])
        prediction = np.asarray([[1.1, 1.8], [4.4, 7.2]])
        cases = [{'condition': 'grf_l25', 'sample_index': 0}]
        condition_data = {'grf_l25': (np.zeros((1, 2, 2, 2)), target[None, ...])}
        cache = {
            (model, 'grf_l25', 0.0): prediction[None, ...]
            for model in ('MSE-M', 'LM-M', 'GM-M')
        }
        panels = _build_case_panels(cases, condition_data, cache, [0.0])
        result = panels['grf_l25'][2]['MSE-M'][0.0]
        expected = 100.0 * np.abs(prediction - target) / np.abs(target)
        self.assertTrue(np.allclose(result['relative_error_percent'], expected))

    def test_runner_scripts_do_not_define_main(self):
        for runner in (
            'run_generate_cases.py',
            'run_preview_samples.py',
            'run_compare_cases.py',
        ):
            source = (PROJECT_ROOT / 'grf_generalization' / runner).read_text(encoding='utf-8')
            self.assertNotIn('def main(', source)
            self.assertNotIn("if __name__ == '__main__'", source)
            self.assertNotIn('if __name__ == "__main__"', source)

    def test_config_separates_main_and_stress_grf_cases(self):
        cfg = get_config(PROJECT_ROOT)
        configured = [case['condition'] for case in cfg['cases']]
        self.assertEqual(configured, ['grf_l20', 'grf_l15', 'grf_l10', 'grf_l8'])
        self.assertEqual(
            [case['condition'] for case in cfg['supplementary_cases']],
            ['grf_l5'],
        )
        self.assertEqual(
            cfg['correlation_lengths_mm'],
            [25.0, 20.0, 15.0, 10.0, 8.0, 5.0],
        )
        self.assertEqual(cfg['noise_levels'], [0, 2, 4, 6, 8, 10])
        self.assertEqual(cfg['output_dir'], PROJECT_ROOT / 'results' / 'grf_ood')
        self.assertEqual(cfg['sample_preview_indices'], list(range(20)))
        self.assertEqual(
            cfg['sample_catalog_conditions'],
            ['grf_l20', 'grf_l15', 'grf_l10', 'grf_l8'],
        )
        self.assertEqual(cfg['stress_preview_conditions'], ['grf_l5'])

    def test_case_outputs_are_routed_to_main_and_supplementary_directories(self):
        cfg = get_config(PROJECT_ROOT)
        conditions = [
            {'condition_id': condition_id}
            for condition_id in ('grf_l20', 'grf_l15', 'grf_l10', 'grf_l8', 'grf_l5')
        ]
        main_cases, supplementary_cases = _validate_cases(cfg, conditions)
        self.assertTrue(all(case['output_group'] == 'cases' for case in main_cases))
        self.assertEqual(
            supplementary_cases[0]['output_group'],
            str(Path('supplementary') / 'cases'),
        )

    def test_preview_indices_are_explicitly_validated(self):
        self.assertEqual(resolve_preview_indices([2, 0, 2], 3), [2, 0])
        with self.assertRaisesRegex(IndexError, 'outside the available range'):
            resolve_preview_indices([3], 3)

    def test_preview_contours_use_the_requested_color_range(self):
        figure, axis = plt.subplots()
        image = _draw_true_field(
            axis,
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            vmin=0.0,
            vmax=10.0,
        )
        self.assertAlmostEqual(float(image.levels[0]), 0.0)
        self.assertAlmostEqual(float(image.levels[-1]), 10.0)
        plt.close(figure)

    def test_grf_is_reproducible_and_shorter_length_is_rougher(self):
        mesh = MeshInfo(9.0, 9.0, 11, 11)
        smooth, _ = generate_grf_fields(mesh, 25.0, 12, seed=123)
        smooth_repeat, _ = generate_grf_fields(mesh, 25.0, 12, seed=123)
        shorter, _ = generate_grf_fields(mesh, 10.0, 12, seed=123)
        below_specimen, _ = generate_grf_fields(mesh, 8.0, 12, seed=123)
        stress, _ = generate_grf_fields(mesh, 5.0, 12, seed=123)

        self.assertTrue(np.array_equal(smooth, smooth_repeat))

        def adjacent_change(fields):
            return np.mean(
                [
                    0.5
                    * (
                        np.mean(np.abs(np.diff(field, axis=0)))
                        + np.mean(np.abs(np.diff(field, axis=1)))
                    )
                    for field in fields
                ]
            )

        self.assertGreater(adjacent_change(shorter), adjacent_change(smooth))
        self.assertGreater(adjacent_change(below_specimen), adjacent_change(shorter))
        self.assertGreater(adjacent_change(stress), adjacent_change(below_specimen))

    def test_steep_fields_remain_continuous_and_bounded(self):
        mesh = MeshInfo(9.0, 9.0, 19, 19)
        fields, metadata = generate_steep_gradient_fields(
            mesh, sample_count=3, seed=456, transition_width_mm=0.75
        )
        self.assertEqual(fields.shape, (3, 20, 20))
        self.assertGreaterEqual(np.min(fields), 1.0)
        self.assertLessEqual(np.max(fields), 8.0)
        self.assertTrue(
            all(row["transition_width_10_90_mm"] == 0.75 for row in metadata)
        )
        self.assertTrue(np.all(np.isfinite(fields)))

    def test_relative_noise_and_metrics(self):
        values = np.arange(1.0, 17.0).reshape(2, 2, 4)
        noisy = apply_relative_noise(values, 6.0, seed=7)
        actual = np.linalg.norm(noisy - values) / np.linalg.norm(values)
        self.assertTrue(np.isclose(actual, 0.06))
        self.assertTrue(np.array_equal(noisy, apply_relative_noise(values, 6.0, seed=7)))

        metrics = field_metrics(values, values + 1.0)
        self.assertTrue(np.isclose(metrics["mae"], 1.0))
        self.assertTrue(np.isclose(metrics["rmse"], 1.0))

    def test_dataset_builder_writes_test_only_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            manifest_path = build_generalization_dataset(
                output_dir,
                sample_count=1,
                steep_count=1,
                correlation_lengths=(25.0, 10.0),
                seed=99,
                nodes_x=6,
                nodes_y=6,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIs(manifest["training_or_validation_use_permitted"], False)
            self.assertEqual(manifest["geometry"]["elements_x"], 5)
            self.assertEqual(
                [row["condition_id"] for row in manifest["conditions"]],
                ["grf_l25", "grf_l10", "steep_sigmoid"],
            )
            self.assertTrue(manifest["sampling_design"]["grf_conditions_are_paired"])
            self.assertEqual(
                manifest["conditions"][0]["seed"], manifest["conditions"][1]["seed"]
            )
            self.assertNotEqual(
                manifest["conditions"][0]["seed"], manifest["conditions"][2]["seed"]
            )
            for condition in manifest["conditions"]:
                condition_dir = output_dir / condition["directory"]
                inputs = loadmat(condition_dir / "input.mat")["U"]
                outputs = loadmat(condition_dir / "output.mat")["E"]
                self.assertEqual(inputs.shape, (1, 2, 6, 6))
                self.assertEqual(outputs.shape, (1, 6, 6))

    def test_paper_table_has_noise_model_rows_and_five_grf_groups(self):
        per_sample = []
        for noise in (0.0, 2.0):
            for model_index, model in enumerate(('MSE-M', 'LM-M', 'GM-M')):
                for condition_index, condition in enumerate(
                    ('grf_l25', 'grf_l20', 'grf_l15', 'grf_l10', 'grf_l8', 'grf_l5')
                ):
                    for sample_id in range(3):
                        per_sample.append(
                            {
                                'model': model,
                                'condition_id': condition,
                                'field_type': 'grf',
                                'distribution_status': 'ID' if condition == 'grf_l25' else 'OOD',
                                'correlation_length_mm': float(condition.split('l')[-1]),
                                'transition_width_10_90_mm': '',
                                'noise_level_percent': noise,
                                'sample_id': sample_id,
                                'relative_l1': 0.01 * (1 + model_index + condition_index + sample_id),
                                'mae': 0.1,
                                'rmse': 0.2,
                            }
                        )
        table = build_paper_table(summarize_metrics(per_sample))
        self.assertEqual(len(table), 6)
        self.assertEqual([row['model'] for row in table[:3]], ['MSE-M', 'LM-M', 'GM-M'])
        for row in table:
            for condition in ('grf_l25', 'grf_l20', 'grf_l15', 'grf_l10', 'grf_l8'):
                self.assertIn(f'{condition}_mean', row)
                self.assertIn(f'{condition}_std', row)
            self.assertNotIn('grf_l5_mean', row)


if __name__ == "__main__":
    unittest.main()

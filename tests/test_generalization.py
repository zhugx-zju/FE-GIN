import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "igfe_unet"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from asm_log.fgm_asm.mesh import MeshInfo
from generalization.data import (
    build_generalization_dataset,
    generate_grf_fields,
    generate_steep_gradient_fields,
)
from generalization.metrics import apply_relative_noise, field_metrics


class GeneralizationTests(unittest.TestCase):
    def test_grf_is_reproducible_and_shorter_length_is_rougher(self):
        mesh = MeshInfo(9.0, 9.0, 11, 11)
        smooth, _ = generate_grf_fields(mesh, 25.0, 12, seed=123)
        smooth_repeat, _ = generate_grf_fields(mesh, 25.0, 12, seed=123)
        shorter, _ = generate_grf_fields(mesh, 10.0, 12, seed=123)

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


if __name__ == "__main__":
    unittest.main()

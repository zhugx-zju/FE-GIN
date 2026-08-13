"""Generate independent smooth-GRF and continuous steep-gradient test cases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import savemat

from asm_log.fgm_asm.fem_forward import fem_assemble, forward_solver
from asm_log.fgm_asm.material import MaterialInfo
from asm_log.fgm_asm.mesh import MeshInfo, setup_boundary_conditions


DEFAULT_CORRELATION_LENGTHS = (25.0, 20.0, 15.0, 10.0)


def _distance_squared(coordinates: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(coordinates**2, axis=1)
    result = squared_norms[:, None] + squared_norms[None, :] - 2.0 * coordinates @ coordinates.T
    result = np.maximum(result, 0.0)
    return 0.5 * (result + result.T)


def _smoothness_metrics(field: np.ndarray, spacing_x: float, spacing_y: float) -> dict[str, float]:
    diff_x = np.diff(field, axis=1)
    diff_y = np.diff(field, axis=0)
    gradient_x = diff_x / spacing_x
    gradient_y = diff_y / spacing_y
    return {
        "mean_adjacent_change": float(0.5 * (np.mean(np.abs(diff_x)) + np.mean(np.abs(diff_y)))),
        "max_nodal_gradient": float(max(np.max(np.abs(gradient_x)), np.max(np.abs(gradient_y)))),
        "field_standard_deviation": float(np.std(field)),
        "field_range": float(np.ptp(field)),
    }


def generate_grf_fields(
    mesh: MeshInfo,
    correlation_length_mm: float,
    sample_count: int,
    seed: int,
    e_maximum: float = 8.0,
    sigma_g: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Reproduce ``GRF_Generate.m`` on a model-compatible nodal grid.

    The covariance is the MATLAB RBF kernel and the Gaussian samples use the
    same tanh-to-[0, 1] mapping. NumPy and MATLAB use different random-number
    streams, so equality for a numeric seed is not claimed.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if correlation_length_mm <= 0:
        raise ValueError("correlation_length_mm must be positive")

    rng = np.random.default_rng(int(seed))
    maxima = np.linspace(1.0, float(e_maximum), int(sample_count))
    maxima = maxima[rng.permutation(sample_count)]
    covariance = np.exp(
        -_distance_squared(mesh.coord) / (2.0 * float(correlation_length_mm) ** 2)
    )
    covariance += 1e-6 * np.eye(mesh.n_nod)
    cholesky = np.linalg.cholesky(covariance)

    fields = np.empty((sample_count, mesh.nods_y, mesh.nods_x), dtype=np.float64)
    metadata = []
    for sample_index in range(sample_count):
        gaussian = cholesky @ rng.standard_normal(mesh.n_nod)
        gaussian = gaussian.reshape(mesh.nods_y, mesh.nods_x)
        normalized = 0.5 * (np.tanh(float(sigma_g) * gaussian) + 1.0)
        field = maxima[sample_index] * normalized
        fields[sample_index] = field
        metadata.append(
            {
                "sample_id": int(sample_index),
                "e_max": float(maxima[sample_index]),
                **_smoothness_metrics(field, mesh.el, mesh.eh),
            }
        )
    return fields, metadata


def generate_steep_gradient_fields(
    mesh: MeshInfo,
    sample_count: int,
    seed: int,
    transition_width_mm: float = 0.75,
    e_minimum: float = 1.0,
    e_maximum: float = 8.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate unseen but continuous sigmoid transitions.

    ``transition_width_mm`` is the distance over which the logistic function
    rises from 10% to 90% of its amplitude. Orientations and interface
    positions vary independently between test samples.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if transition_width_mm <= 0:
        raise ValueError("transition_width_mm must be positive")

    rng = np.random.default_rng(int(seed))
    maxima = np.linspace(max(e_minimum + 0.5, 2.0), float(e_maximum), sample_count)
    maxima = maxima[rng.permutation(sample_count)]
    scale = float(transition_width_mm) / (2.0 * np.log(9.0))
    centered_x = mesh.plot_x - 0.5 * mesh.geo_l
    centered_y = mesh.plot_y - 0.5 * mesh.geo_h

    fields = np.empty((sample_count, mesh.nods_y, mesh.nods_x), dtype=np.float64)
    metadata = []
    for sample_index in range(sample_count):
        angle = rng.uniform(0.0, np.pi)
        offset = rng.uniform(-0.12, 0.12) * min(mesh.geo_l, mesh.geo_h)
        signed_distance = (
            np.cos(angle) * centered_x + np.sin(angle) * centered_y - offset
        )
        sigmoid = 1.0 / (1.0 + np.exp(-signed_distance / scale))
        field = float(e_minimum) + (maxima[sample_index] - float(e_minimum)) * sigmoid
        fields[sample_index] = field
        metadata.append(
            {
                "sample_id": int(sample_index),
                "e_min": float(e_minimum),
                "e_max": float(maxima[sample_index]),
                "orientation_deg": float(np.degrees(angle)),
                "interface_offset_mm": float(offset),
                "transition_width_10_90_mm": float(transition_width_mm),
                **_smoothness_metrics(field, mesh.el, mesh.eh),
            }
        )
    return fields, metadata


def solve_displacements(
    fields: np.ndarray,
    mesh: MeshInfo,
    poisson_ratio: float = 0.3,
    total_force: float = 0.01,
) -> np.ndarray:
    """Solve the shared plane-stress forward problem for each modulus field."""
    boundary = setup_boundary_conditions(mesh, mesh.geo_l, mesh.geo_h, total_force)
    inputs = np.empty((len(fields), 2, mesh.nods_y, mesh.nods_x), dtype=np.float64)
    for sample_index, field in enumerate(fields):
        material = MaterialInfo(nu=float(poisson_ratio), dis_type="grf")
        material.update(np.asarray(field, dtype=np.float64).reshape(-1))
        fem_info = fem_assemble(mesh, material, boundary)
        displacement = forward_solver(fem_info)
        inputs[sample_index, 0] = displacement[0::2].reshape(mesh.nods_y, mesh.nods_x)
        inputs[sample_index, 1] = displacement[1::2].reshape(mesh.nods_y, mesh.nods_x)
    return inputs


def _write_condition(
    directory: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, float]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve the variable names and tensor layout used by the training data.
    savemat(directory / "input.mat", {"U": inputs.astype(np.float32)})
    savemat(directory / "output.mat", {"E": targets.astype(np.float32)})
    with (directory / "samples.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_generalization_dataset(
    output_root: str | Path,
    sample_count: int = 20,
    steep_count: int | None = None,
    correlation_lengths: tuple[float, ...] = DEFAULT_CORRELATION_LENGTHS,
    seed: int = 8606,
    transition_width_mm: float = 0.75,
    nodes_x: int = 40,
    nodes_y: int = 40,
    include_steep_gradient: bool = True,
) -> Path:
    """Generate all test-only cases and return the manifest path."""
    if nodes_x < 2 or nodes_y < 2:
        raise ValueError("At least two nodes are required in each direction")
    steep_count = sample_count if steep_count is None else int(steep_count)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mesh = MeshInfo(9.0, 9.0, nodes_x - 1, nodes_y - 1)

    conditions = []
    # Reuse the same independent test seed for every ell. This gives paired
    # samples: latent normal draws and E_max ordering are held fixed while
    # only the RBF covariance length changes.
    grf_seed = int(seed)
    for ell in correlation_lengths:
        condition_id = f"grf_l{float(ell):g}".replace(".", "p")
        fields, metadata = generate_grf_fields(
            mesh, float(ell), int(sample_count), grf_seed
        )
        inputs = solve_displacements(fields, mesh)
        condition_dir = output_root / condition_id
        _write_condition(condition_dir, inputs, fields, metadata)
        conditions.append(
            {
                "condition_id": condition_id,
                "field_type": "grf",
                "correlation_length_mm": float(ell),
                "distribution_status": "ID" if np.isclose(float(ell), 25.0) else "OOD",
                "sample_count": int(sample_count),
                "seed": grf_seed,
                "paired_grf_samples": True,
                "directory": condition_id,
            }
        )

    steep_seed = int(seed) + 1000
    if include_steep_gradient:
        steep_fields, steep_metadata = generate_steep_gradient_fields(
            mesh,
            steep_count,
            steep_seed,
            transition_width_mm=float(transition_width_mm),
        )
        steep_inputs = solve_displacements(steep_fields, mesh)
        steep_id = "steep_sigmoid"
        _write_condition(output_root / steep_id, steep_inputs, steep_fields, steep_metadata)
        conditions.append(
            {
                "condition_id": steep_id,
                "field_type": "continuous_sigmoid",
                "transition_width_10_90_mm": float(transition_width_mm),
                "distribution_status": "OOD",
                "sample_count": int(steep_count),
                "seed": steep_seed,
                "directory": steep_id,
            }
        )

    manifest = {
        "schema_version": 1,
        "purpose": "test_only_zero_shot_generalization",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_or_validation_use_permitted": False,
        "base_seed": int(seed),
        "sampling_design": {
            "grf_conditions_are_paired": True,
            "grf_pairing_variables_held_fixed": ["standard_normal_draws", "e_max_order"],
            "variable_changed_between_grf_conditions": "correlation_length_mm",
            "steep_gradient_included": bool(include_steep_gradient),
            "steep_gradient_uses_separate_seed": bool(include_steep_gradient),
        },
        "geometry": {
            "width_mm": 9.0,
            "height_mm": 9.0,
            "nodes_x": int(nodes_x),
            "nodes_y": int(nodes_y),
            "elements_x": int(nodes_x - 1),
            "elements_y": int(nodes_y - 1),
        },
        "forward_problem": {
            "plane_stress": True,
            "poisson_ratio": 0.3,
            "total_right_edge_force": 0.01,
        },
        "grf_definition": {
            "covariance": "exp(-distance_squared / (2 * ell^2))",
            "diagonal_jitter": 1e-6,
            "sigma_g": 1.0,
            "mapping": "E_max * (tanh(sigma_g * g) + 1) / 2",
            "source_compatibility": "data_generation/uniform_pressure_load/GRF_Generate.m",
        },
        "conditions": conditions,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    manifest["files"] = {
        str(path.relative_to(output_root)).replace("\\", "/"): _sha256(path)
        for path in sorted(output_root.glob("*/*"))
        if path.is_file()
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path

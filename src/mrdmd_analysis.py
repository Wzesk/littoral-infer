"""Multi-resolution Dynamic Mode Decomposition for SDF shoreline analysis.

Supports decomposition on either raw SDF snapshots or a reduced set of
regularized SDF coefficients (recommended). The reduced-basis path is
intended for long-run modal analysis after temporal SDF hindcasting has
produced a regular time grid.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import torch

try:
    from pydmd import DMD, MrDMD
except ImportError:
    MrDMD = None


def resolve_site_sdf_folder(site_path: str, preferred: str = "shoreline") -> str:
    """Resolve the preferred SDF folder for a processed site directory."""
    candidates = []
    if preferred == "shoreline":
        candidates.extend(
            [
                os.path.join(site_path, "SDF_INFERENCE"),
                os.path.join(site_path, "SDF_FROM_SHORELINE"),
            ]
        )
    elif preferred == "direct":
        candidates.extend(
            [
                os.path.join(site_path, "SDF_DIRECT_MODELS"),
                os.path.join(site_path, "SDF_MODELS"),
            ]
        )
    candidates.extend(
        [
            os.path.join(site_path, "SDF_INFERENCE"),
            os.path.join(site_path, "SDF_FROM_SHORELINE"),
            os.path.join(site_path, "SDF_DIRECT_MODELS"),
            os.path.join(site_path, "SDF_MODELS"),
        ]
    )
    for folder in candidates:
        if os.path.isdir(folder) and any(name.endswith(".pth") for name in os.listdir(folder)):
            return folder
    raise FileNotFoundError(f"No SDF checkpoint folder found under {site_path}")


def _sdf_to_snapshot(sdf_path: str, grid_res: int, device: str = "cpu") -> np.ndarray:
    """Load a ShapeSDF .pth checkpoint and evaluate it on a fixed grid."""
    import sys

    refine_dir = os.path.join(os.path.dirname(__file__), "..", "..", "littoral_refine")
    if refine_dir not in sys.path:
        sys.path.insert(0, refine_dir)
    from sdf_shoreline import IGRNetwork

    checkpoint = torch.load(sdf_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        conf = checkpoint["config"]
        h_dim = conf.get("hidden_dim", 256)
        n_lay = conf.get("num_layers", 4)
    else:
        h_dim, n_lay = 256, 4

    model = IGRNetwork(2, 1, h_dim, n_lay).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    coords = np.linspace(-1, 1, grid_res)
    X, Y = np.meshgrid(coords, coords)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        sdf_vals = model(grid_t).squeeze().cpu().numpy()
    return sdf_vals.astype(np.float32)


def _smooth_time_series(matrix: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or matrix.shape[1] < 3:
        return matrix
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(matrix)
    for row_idx in range(matrix.shape[0]):
        padded = np.pad(matrix[row_idx], (pad, pad), mode="edge")
        smoothed[row_idx] = np.convolve(padded, kernel, mode="valid")
    return smoothed


class ShorelineMrDMD:
    """Multi-resolution DMD decomposition of a shoreline SDF time series."""

    def __init__(
        self,
        max_level: int = 6,
        max_cycles: int = 1,
        svd_rank: int = -1,
        *,
        default_feature_rank: Optional[int] = None,
        default_energy_threshold: Optional[float] = 0.995,
        default_coefficient_smoothing: int = 1,
    ):
        if MrDMD is None:
            raise ImportError("pydmd is required. Install with: pip install pydmd")
        self.max_level = max_level
        self.max_cycles = max_cycles
        self.svd_rank = svd_rank
        self.default_feature_rank = default_feature_rank
        self.default_energy_threshold = default_energy_threshold
        self.default_coefficient_smoothing = int(default_coefficient_smoothing)

        self._dmd: Optional[MrDMD] = None
        self._data_matrix: Optional[np.ndarray] = None
        self._norm_min: float = 0.0
        self._norm_range: float = 1.0
        self._grid_res: int = 64
        self._file_order: list[str] = []
        self._dt: float = 1.0

        self._spatial_mean: Optional[np.ndarray] = None
        self._basis: Optional[np.ndarray] = None
        self._feature_matrix: Optional[np.ndarray] = None
        self._feature_rank: Optional[int] = None
        self._energy_threshold: Optional[float] = None
        self._coefficient_smoothing: int = 1

    def _choose_rank(self, singular_values: np.ndarray, feature_rank: Optional[int], energy_threshold: Optional[float]) -> Optional[int]:
        if feature_rank is not None:
            return max(1, min(int(feature_rank), len(singular_values)))
        if energy_threshold is None:
            return None
        energy = np.cumsum(singular_values ** 2) / max(np.sum(singular_values ** 2), 1e-12)
        return int(np.searchsorted(energy, energy_threshold) + 1)

    def _encode_matrix(
        self,
        data_matrix: np.ndarray,
        *,
        feature_rank: Optional[int],
        energy_threshold: Optional[float],
        coefficient_smoothing: int,
    ) -> np.ndarray:
        self._spatial_mean = data_matrix.mean(axis=1, keepdims=True).astype(np.float32)
        centered = (data_matrix - self._spatial_mean).astype(np.float32)

        U, s, _ = np.linalg.svd(centered, full_matrices=False)
        rank = self._choose_rank(s, feature_rank, energy_threshold)
        self._feature_rank = rank
        self._energy_threshold = energy_threshold
        self._coefficient_smoothing = int(coefficient_smoothing)

        if rank is None or rank >= min(centered.shape):
            self._basis = None
            features = centered
        else:
            self._basis = U[:, :rank].astype(np.float32)
            features = (self._basis.T @ centered).astype(np.float32)

        features = _smooth_time_series(features, self._coefficient_smoothing)
        self._feature_matrix = features
        return features

    def _decode_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if self._basis is None:
            return matrix + self._spatial_mean
        return (self._basis @ matrix) + self._spatial_mean

    def fit(
        self,
        data_matrix: np.ndarray,
        dt: float = 1.0,
        *,
        feature_rank: Optional[int] = None,
        energy_threshold: Optional[float] = None,
        coefficient_smoothing: Optional[int] = None,
    ) -> "ShorelineMrDMD":
        self._dt = dt
        use_rank = self.default_feature_rank if feature_rank is None else feature_rank
        use_energy = self.default_energy_threshold if energy_threshold is None else energy_threshold
        use_smoothing = self.default_coefficient_smoothing if coefficient_smoothing is None else coefficient_smoothing

        working = self._encode_matrix(
            np.asarray(data_matrix, dtype=np.float32),
            feature_rank=use_rank,
            energy_threshold=use_energy,
            coefficient_smoothing=use_smoothing,
        )

        self._norm_min = float(working.min())
        self._norm_range = float(working.max() - self._norm_min)
        if self._norm_range < 1e-12:
            self._norm_range = 1.0
        norm = (working - self._norm_min) / self._norm_range * 2 - 1
        self._data_matrix = norm

        sub_dmd = DMD(svd_rank=self.svd_rank)
        self._dmd = MrDMD(sub_dmd, max_level=self.max_level, max_cycles=self.max_cycles)
        self._dmd.fit(X=norm)
        return self

    def fit_from_sdf_folder(
        self,
        folder: str,
        *,
        grid_res: int = 64,
        device: str = "cpu",
        dt: float = 1.0,
        feature_rank: Optional[int] = 16,
        energy_threshold: Optional[float] = 0.995,
        coefficient_smoothing: int = 1,
    ) -> "ShorelineMrDMD":
        files = sorted(glob.glob(os.path.join(folder, "*.pth")))
        if len(files) < 3:
            raise ValueError(f"Need >=3 SDF checkpoints, found {len(files)} in {folder}")

        self._file_order = [os.path.basename(path) for path in files]
        self._grid_res = grid_res
        snapshots = [_sdf_to_snapshot(path, grid_res, device) for path in files]
        data_matrix = np.column_stack(snapshots)
        return self.fit(
            data_matrix,
            dt=dt,
            feature_rank=feature_rank,
            energy_threshold=energy_threshold,
            coefficient_smoothing=coefficient_smoothing,
        )

    def fit_from_site(
        self,
        site_path: str,
        *,
        preferred: str = "shoreline",
        grid_res: int = 64,
        device: str = "cpu",
        dt: float = 1.0,
        feature_rank: Optional[int] = 16,
        energy_threshold: Optional[float] = 0.995,
        coefficient_smoothing: int = 1,
    ) -> "ShorelineMrDMD":
        folder = resolve_site_sdf_folder(site_path, preferred=preferred)
        return self.fit_from_sdf_folder(
            folder,
            grid_res=grid_res,
            device=device,
            dt=dt,
            feature_rank=feature_rank,
            energy_threshold=energy_threshold,
            coefficient_smoothing=coefficient_smoothing,
        )

    def fit_from_temporal_sdf(
        self,
        temporal_model,
        days: list[float],
        *,
        grid_res: int = 64,
        dt: float = 1.0,
        feature_rank: Optional[int] = 16,
        energy_threshold: Optional[float] = 0.995,
        coefficient_smoothing: int = 3,
    ) -> "ShorelineMrDMD":
        self._grid_res = grid_res
        self._file_order = [f"day_{float(day):.3f}" for day in days]
        snapshots = [
            temporal_model.evaluate_sdf_grid(day, grid_res=grid_res).reshape(-1).astype(np.float32)
            for day in days
        ]
        data_matrix = np.column_stack(snapshots)
        return self.fit(
            data_matrix,
            dt=dt,
            feature_rank=feature_rank,
            energy_threshold=energy_threshold,
            coefficient_smoothing=coefficient_smoothing,
        )

    @property
    def n_levels(self) -> int:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        return self.max_level

    def reconstructed_data(self, denorm: bool = True) -> np.ndarray:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        rec = np.real(self._dmd.reconstructed_data)
        if denorm:
            rec = (rec + 1) / 2 * self._norm_range + self._norm_min
        return self._decode_matrix(rec)

    def get_level_contribution(self, level: int, denorm: bool = False) -> np.ndarray:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        partial = np.real(self._dmd.partial_reconstructed_data(level=level))
        if denorm:
            partial = (partial + 1) / 2 * self._norm_range + self._norm_min
        return self._decode_matrix(partial)

    def cumulative_reconstruction(self, up_to_level: int, denorm: bool = True) -> np.ndarray:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        total = self._dmd.partial_reconstructed_data(level=0)
        for level in range(1, up_to_level + 1):
            total = total + self._dmd.partial_reconstructed_data(level=level)
        total = np.real(total)
        if denorm:
            total = (total + 1) / 2 * self._norm_range + self._norm_min
        return self._decode_matrix(total)

    def eigenvalues(self) -> np.ndarray:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        return self._dmd.eigs

    def forecast(self, n_steps: int, denorm: bool = True) -> np.ndarray:
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")

        n_train = self._data_matrix.shape[1]
        modes = self._dmd.modes
        eigs = self._dmd.eigs
        dynamics = self._dmd.dynamics
        b0 = dynamics[:, 0]

        future_dynamics = np.zeros((len(eigs), n_steps), dtype=complex)
        for idx in range(n_steps):
            t_step = n_train + idx
            future_dynamics[:, idx] = b0 * (eigs ** t_step)

        forecast_data = np.real(modes @ future_dynamics)
        if denorm:
            forecast_data = (forecast_data + 1) / 2 * self._norm_range + self._norm_min
        return self._decode_matrix(forecast_data)

    def feature_coefficients(self, denorm: bool = True) -> np.ndarray:
        if self._feature_matrix is None:
            raise RuntimeError("Call fit() first.")
        if denorm:
            return self._feature_matrix.copy()
        return self._data_matrix.copy()

    def save(self, filepath: str) -> None:
        if self._dmd is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        np.savez_compressed(
            filepath,
            data_matrix=self._data_matrix,
            norm_min=self._norm_min,
            norm_range=self._norm_range,
            grid_res=self._grid_res,
            dt=self._dt,
            max_level=self.max_level,
            max_cycles=self.max_cycles,
            svd_rank=self.svd_rank,
            file_order=np.array(self._file_order, dtype=object),
            spatial_mean=self._spatial_mean,
            basis=self._basis,
            feature_matrix=self._feature_matrix,
            feature_rank=-1 if self._feature_rank is None else int(self._feature_rank),
            energy_threshold=-1.0 if self._energy_threshold is None else float(self._energy_threshold),
            coefficient_smoothing=int(self._coefficient_smoothing),
            default_feature_rank=-1 if self.default_feature_rank is None else int(self.default_feature_rank),
            default_energy_threshold=-1.0 if self.default_energy_threshold is None else float(self.default_energy_threshold),
            default_coefficient_smoothing=int(self.default_coefficient_smoothing),
        )

    @classmethod
    def load(cls, filepath: str) -> "ShorelineMrDMD":
        data = np.load(filepath, allow_pickle=True)
        obj = cls(
            max_level=int(data["max_level"]),
            max_cycles=int(data["max_cycles"]),
            svd_rank=int(data["svd_rank"]),
            default_feature_rank=None if int(data["default_feature_rank"]) < 0 else int(data["default_feature_rank"]),
            default_energy_threshold=None if float(data["default_energy_threshold"]) < 0 else float(data["default_energy_threshold"]),
            default_coefficient_smoothing=int(data["default_coefficient_smoothing"]),
        )
        obj._grid_res = int(data["grid_res"])
        obj._file_order = list(data["file_order"])
        obj._spatial_mean = data["spatial_mean"]
        obj._basis = data["basis"] if data["basis"].size else None
        obj._feature_matrix = data["feature_matrix"]
        obj._feature_rank = None if int(data["feature_rank"]) < 0 else int(data["feature_rank"])
        obj._energy_threshold = None if float(data["energy_threshold"]) < 0 else float(data["energy_threshold"])
        obj._coefficient_smoothing = int(data["coefficient_smoothing"])
        obj.fit(
            obj._decode_matrix(obj._feature_matrix),
            dt=float(data["dt"]),
            feature_rank=obj._feature_rank,
            energy_threshold=obj._energy_threshold,
            coefficient_smoothing=obj._coefficient_smoothing,
        )
        return obj

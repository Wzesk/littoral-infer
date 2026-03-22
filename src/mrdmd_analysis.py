"""Multi-resolution Dynamic Mode Decomposition for SDF shoreline analysis.

Decomposes the spatiotemporal SDF field into slow (trend) and fast
(seasonal / transient) components using the PyDMD MrDMD implementation.
Provides reconstruction, level-based decomposition, and time-extrapolation
(forecasting) of the SDF field.

Typical usage
-------------
>>> from mrdmd_analysis import ShorelineMrDMD
>>> mrdmd = ShorelineMrDMD(max_level=6)
>>> mrdmd.fit_from_sdf_folder("path/to/sdf_pth_files", grid_res=64)
>>> trend = mrdmd.get_level_contribution(0)        # slow drift
>>> future = mrdmd.forecast(n_steps=10)             # extrapolate
>>> mrdmd.save("decomposition.npz")
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import torch

try:
    from pydmd import MrDMD, DMD
except ImportError:
    MrDMD = None  # graceful degrade — error raised at fit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sdf_to_snapshot(sdf_path: str, grid_res: int, device: str = "cpu") -> np.ndarray:
    """Load a ShapeSDF .pth checkpoint and evaluate it on a fixed grid.

    Returns a 1-D array of length ``grid_res**2`` (row-major flattened SDF
    values).
    """
    # Import IGRNetwork lazily to avoid hard dependency at module level
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
    return sdf_vals


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShorelineMrDMD:
    """Multi-resolution DMD decomposition of a shoreline SDF time series.

    Parameters
    ----------
    max_level : int
        Number of hierarchical levels for MrDMD.
    max_cycles : int
        Maximum number of oscillation cycles kept per level.
    svd_rank : int
        Truncation rank for the inner DMD (-1 = no truncation).
    """

    def __init__(self, max_level: int = 6, max_cycles: int = 1, svd_rank: int = -1):
        if MrDMD is None:
            raise ImportError("pydmd is required.  Install with: pip install pydmd")
        self.max_level = max_level
        self.max_cycles = max_cycles
        self.svd_rank = svd_rank

        # Populated after fit()
        self._dmd: Optional[MrDMD] = None
        self._data_matrix: Optional[np.ndarray] = None
        self._norm_min: float = 0.0
        self._norm_range: float = 1.0
        self._grid_res: int = 64
        self._file_order: list[str] = []
        self._dt: float = 1.0  # time step between snapshots

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data_matrix: np.ndarray, dt: float = 1.0) -> "ShorelineMrDMD":
        """Fit MrDMD on an already-assembled data matrix.

        Parameters
        ----------
        data_matrix : np.ndarray
            Shape ``(n_spatial, n_snapshots)`` — each column is one time
            snapshot of the flattened SDF field.
        dt : float
            Time step between successive snapshots.

        Returns
        -------
        self
        """
        self._dt = dt
        # Normalise to [-1, 1] for numeric stability
        self._norm_min = float(data_matrix.min())
        self._norm_range = float(data_matrix.max() - data_matrix.min())
        if self._norm_range < 1e-12:
            self._norm_range = 1.0
        norm = (data_matrix - self._norm_min) / self._norm_range * 2 - 1
        self._data_matrix = norm

        sub_dmd = DMD(svd_rank=self.svd_rank)
        self._dmd = MrDMD(sub_dmd, max_level=self.max_level, max_cycles=self.max_cycles)
        self._dmd.fit(X=norm)
        return self

    def fit_from_sdf_folder(
        self,
        folder: str,
        grid_res: int = 64,
        device: str = "cpu",
        dt: float = 1.0,
    ) -> "ShorelineMrDMD":
        """Load SDF ``.pth`` checkpoints, build snapshot matrix, and fit.

        Files are sorted lexicographically — file names should encode
        chronological order (e.g. ``day_001.pth``).

        Parameters
        ----------
        folder : str
            Directory containing ``.pth`` SDF checkpoints.
        grid_res : int
            Resolution of the evaluation grid (``grid_res × grid_res``).
        device : str
            Torch device for SDF network evaluation.
        dt : float
            Time step between successive snapshots.
        """
        files = sorted(glob.glob(os.path.join(folder, "*.pth")))
        if len(files) < 3:
            raise ValueError(f"Need ≥3 SDF checkpoints, found {len(files)} in {folder}")

        self._file_order = [os.path.basename(f) for f in files]
        self._grid_res = grid_res

        snapshots = []
        for f in files:
            snap = _sdf_to_snapshot(f, grid_res, device)
            snapshots.append(snap)

        data_matrix = np.column_stack(snapshots)  # (n_spatial, n_time)
        return self.fit(data_matrix, dt=dt)

    # ------------------------------------------------------------------
    # Decomposition queries
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Number of decomposition levels."""
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        return self.max_level

    def reconstructed_data(self, denorm: bool = True) -> np.ndarray:
        """Full MrDMD reconstruction ``(n_spatial, n_time)``.

        Parameters
        ----------
        denorm : bool
            If True (default), map back from [-1,1] to original value range.
        """
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        rec = np.real(self._dmd.reconstructed_data)
        if denorm:
            rec = (rec + 1) / 2 * self._norm_range + self._norm_min
        return rec

    def get_level_contribution(self, level: int, denorm: bool = False) -> np.ndarray:
        """Partial reconstruction from a single decomposition level.

        Level 0 captures the slowest dynamics (long-term trend).
        Higher levels capture progressively faster oscillations.

        Returns shape ``(n_spatial, n_time)``.
        """
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        partial = np.real(self._dmd.partial_reconstructed_data(level=level))
        if denorm:
            # Level contributions are additive — only denorm the full sum
            partial = (partial + 1) / 2 * self._norm_range + self._norm_min
        return partial

    def cumulative_reconstruction(self, up_to_level: int, denorm: bool = True) -> np.ndarray:
        """Sum of levels 0 … *up_to_level* (inclusive).

        Useful for examining how much detail each level adds.
        """
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        total = self._dmd.partial_reconstructed_data(level=0)
        for lv in range(1, up_to_level + 1):
            total = total + self._dmd.partial_reconstructed_data(level=lv)
        total = np.real(total)
        if denorm:
            total = (total + 1) / 2 * self._norm_range + self._norm_min
        return total

    def eigenvalues(self) -> np.ndarray:
        """All eigenvalues across levels."""
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")
        return self._dmd.eigs

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(self, n_steps: int, denorm: bool = True) -> np.ndarray:
        """Extrapolate the SDF field forward in time.

        Uses the DMD eigenvalues and modes to project the dynamics beyond
        the training window.  Works best for the slow (low-level) modes;
        high-frequency levels may oscillate unrealistically.

        Parameters
        ----------
        n_steps : int
            Number of future time steps to generate.
        denorm : bool
            Map back to original value range.

        Returns
        -------
        np.ndarray
            Shape ``(n_spatial, n_steps)``.
        """
        if self._dmd is None:
            raise RuntimeError("Call fit() first.")

        n_train = self._data_matrix.shape[1]
        modes = self._dmd.modes        # (n_spatial, n_modes)
        eigs = self._dmd.eigs          # (n_modes,)
        dynamics = self._dmd.dynamics  # (n_modes, n_time)

        # Initial amplitudes from the first dynamics column
        b0 = dynamics[:, 0]

        # Extrapolate: x(t) = Φ · diag(b0 · λ^t)
        future_dynamics = np.zeros((len(eigs), n_steps), dtype=complex)
        for i in range(n_steps):
            t_step = n_train + i
            future_dynamics[:, i] = b0 * (eigs ** t_step)

        forecast_data = np.real(modes @ future_dynamics)
        if denorm:
            forecast_data = (forecast_data + 1) / 2 * self._norm_range + self._norm_min
        return forecast_data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save decomposition results to a compressed ``.npz`` file."""
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
        )

    @classmethod
    def load(cls, filepath: str) -> "ShorelineMrDMD":
        """Load and re-fit from a saved ``.npz`` file."""
        d = np.load(filepath, allow_pickle=True)
        obj = cls(
            max_level=int(d["max_level"]),
            max_cycles=int(d["max_cycles"]),
            svd_rank=int(d["svd_rank"]),
        )
        obj._grid_res = int(d["grid_res"])
        obj._file_order = list(d["file_order"])
        obj.fit(d["data_matrix"], dt=float(d["dt"]))
        return obj

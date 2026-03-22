"""
3D Temporal SDF — modularised ShorelineEvolutionSDF.

Maps (x, y, t) → SDF value using a neural implicit field.  The zero
level-set at any time slice gives the shoreline for that day.

Key features:
- Spatial Eikonal constraint (|∇_{xy} φ| ≈ 1) for valid distance fields
- Temporal gradient is *free* (no Eikonal on t) — allows flexible evolution
- Save / load serialisation
- ``fit_from_sdf_folder`` convenience: loads 2D ShapeSDF .pth files,
  extracts boundary points, trains the 3D field
- ``get_shoreline_at_day`` / ``predict_shoreline`` for interpolation
  and extrapolation

Usage
-----
    from temporal_sdf import ShorelineEvolutionSDF

    model = ShorelineEvolutionSDF(hidden_dim=256, num_layers=6, device='cuda')
    model.fit_from_sdf_folder('/path/to/SDF_FILTERED', days=[0, 30, 60, ...])
    pts_day45 = model.get_shoreline_at_day(45, n_points=200)
    model.save('temporal_sdf.pth')
"""

import glob
import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import measure
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TemporalIGRNetwork(nn.Module):
    """MLP mapping (x, y, t) → SDF value."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1,
                 hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        layers: list = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Softplus(beta=100))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus(beta=100))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShorelineEvolutionSDF:
    """Neural implicit (x, y, t) → SDF for temporal shoreline modelling."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        device: str = "cpu",
    ):
        self.device = device
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        self.model = TemporalIGRNetwork(
            in_dim=3, out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.is_fitted = False

        # Normalisation for spatial + temporal dimensions
        self.spatial_center: Optional[np.ndarray] = None
        self.spatial_scale: Optional[float] = None
        self.t_min: Optional[float] = None
        self.t_max: Optional[float] = None

    # ── Fitting ───────────────────────────────────────────────────────

    def fit(
        self,
        boundary_points_per_day: List[np.ndarray],
        days: List[float],
        *,
        iterations: int = 3000,
        lr: float = 1e-3,
        lambda_eik: float = 0.1,
        n_rand_ratio: int = 2,
    ):
        """Train the 3D SDF from boundary point sets at known days.

        Parameters
        ----------
        boundary_points_per_day : list[np.ndarray]
            Each entry is ``(N_i, 2)`` boundary points for day ``days[i]``.
        days : list[float]
            Day offsets (e.g. 0, 30, 60, …) corresponding to each point set.
        iterations : int
            Training iterations.
        lambda_eik : float
            Eikonal loss weight (spatial only).
        n_rand_ratio : int
            Random sample points per boundary point.
        """
        assert len(boundary_points_per_day) == len(days)

        # Concatenate all boundary points + assign normalised t
        all_pts = np.concatenate(boundary_points_per_day, axis=0)
        self.spatial_center = all_pts.mean(axis=0)
        self.spatial_scale = float(np.max(np.linalg.norm(
            all_pts - self.spatial_center, axis=1
        ))) * 1.1

        self.t_min = float(min(days))
        self.t_max = float(max(days))
        t_range = max(self.t_max - self.t_min, 1.0)

        # Build (x_norm, y_norm, t_norm) manifold tensor
        manifold_list = []
        for pts, day in zip(boundary_points_per_day, days):
            norm_xy = (pts - self.spatial_center) / self.spatial_scale
            t_col = np.full((len(pts), 1), (day - self.t_min) / t_range)
            manifold_list.append(np.hstack([norm_xy, t_col]))

        manifold = torch.tensor(
            np.concatenate(manifold_list, axis=0),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        n_manifold = manifold.shape[0]

        logger.info("Training 3D SDF on %d manifold points over %d days…",
                     n_manifold, len(days))

        for it in range(iterations):
            optimizer.zero_grad()

            # Manifold loss: f(x, y, t) = 0 on boundary
            pred = self.model(manifold)
            loss_manifold = pred.abs().mean()

            # Random domain points for Eikonal
            n_rand = n_manifold * n_rand_ratio
            xy_rand = (torch.rand(n_rand, 2, device=self.device) * 2 - 1)
            t_rand = torch.rand(n_rand, 1, device=self.device)
            rand_pts = torch.cat([xy_rand, t_rand], dim=1).requires_grad_(True)

            all_pts_t = torch.cat([manifold, rand_pts], dim=0)
            y = self.model(all_pts_t)

            grads = torch.autograd.grad(
                outputs=y, inputs=all_pts_t,
                grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True,
            )[0]

            # Spatial Eikonal only (first 2 dims)
            spatial_grad = grads[:, :2]
            grad_norm = spatial_grad.norm(2, dim=1)
            loss_eikonal = ((grad_norm - 1) ** 2).mean()

            loss = loss_manifold + lambda_eik * loss_eikonal
            loss.backward()
            optimizer.step()

            if (it + 1) % 500 == 0:
                logger.info(
                    "  iter %d/%d  manifold=%.5f  eikonal=%.5f",
                    it + 1, iterations, loss_manifold.item(), loss_eikonal.item(),
                )

        self.is_fitted = True
        logger.info("3D SDF training complete.")

    def fit_from_sdf_folder(
        self,
        sdf_folder: str,
        days: Optional[List[float]] = None,
        *,
        n_boundary_points: int = 300,
        **fit_kwargs,
    ):
        """Convenience: load 2D ShapeSDF ``.pth`` files and train 3D SDF.

        Parameters
        ----------
        sdf_folder : str
            Directory containing ``.pth`` ShapeSDF checkpoints.
        days : list[float] or None
            Day offsets per file (sorted order).  If ``None``, uses
            0, 1, 2, … as indices.
        n_boundary_points : int
            Points to extract per 2D SDF boundary.
        """
        import sys
        # Try to import ShapeSDF from littoral_refine
        refine_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'littoral_refine'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'littoral_refine'),
        ]
        for p in refine_candidates:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        from sdf_shoreline import ShapeSDF

        pth_files = sorted(glob.glob(os.path.join(sdf_folder, "*.pth")))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {sdf_folder}")

        if days is None:
            days = list(range(len(pth_files)))

        assert len(days) == len(pth_files), \
            f"days ({len(days)}) must match pth files ({len(pth_files)})"

        boundary_sets: List[np.ndarray] = []
        for f in pth_files:
            sdf = ShapeSDF(device=self.device)
            sdf.load(f)
            pts = sdf.get_boundary_points(n_boundary_points)
            boundary_sets.append(pts)

        self.fit(boundary_sets, days, **fit_kwargs)

    # ── Inference ─────────────────────────────────────────────────────

    def get_shoreline_at_day(
        self,
        day: float,
        *,
        n_points: int = 200,
        grid_res: int = 256,
    ) -> np.ndarray:
        """Extract the zero level-set at a given day.

        Parameters
        ----------
        day : float
            Day offset (same scale as training ``days``).
        n_points : int
            Number of evenly-spaced boundary points to return.
        grid_res : int
            Grid resolution for marching squares.

        Returns
        -------
        np.ndarray
            ``(n_points, 2)`` world-coordinate boundary points.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        t_range = max(self.t_max - self.t_min, 1.0)
        t_norm = (day - self.t_min) / t_range

        # Build 2D grid at this time slice
        x = np.linspace(-1.1, 1.1, grid_res)
        y = np.linspace(-1.1, 1.1, grid_res)
        X, Y = np.meshgrid(x, y)
        grid_xy = np.column_stack([X.ravel(), Y.ravel()])
        t_col = np.full((grid_xy.shape[0], 1), t_norm)
        grid_xyt = np.hstack([grid_xy, t_col])

        with torch.no_grad():
            t_in = torch.tensor(grid_xyt, dtype=torch.float32, device=self.device)
            Z = self.model(t_in).cpu().numpy().reshape(grid_res, grid_res)

        contours = measure.find_contours(Z, 0.0)
        if not contours:
            raise ValueError(f"No zero level-set at day {day}")

        contour = max(contours, key=len)

        # Convert grid coords → normalised → world
        r, c = contour[:, 0], contour[:, 1]
        x_n = -1.1 + (c / (grid_res - 1)) * 2.2
        y_n = -1.1 + (r / (grid_res - 1)) * 2.2
        norm_pts = np.column_stack([x_n, y_n])
        world_pts = norm_pts * self.spatial_scale + self.spatial_center

        # Resample evenly
        dists = np.linalg.norm(np.diff(world_pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(dists)])
        if cum[-1] == 0:
            return world_pts[:n_points]
        interp = interp1d(cum, world_pts, axis=0, kind="linear")
        targets = np.linspace(0, cum[-1], n_points, endpoint=False)
        return interp(targets)

    def predict_shoreline(
        self,
        future_day: float,
        n_points: int = 200,
    ) -> np.ndarray:
        """Convenience alias for ``get_shoreline_at_day`` (extrapolation)."""
        return self.get_shoreline_at_day(future_day, n_points=n_points)

    # ── Serialisation ─────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save model, config, and normalisation to a file."""
        state = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "is_fitted": self.is_fitted,
            "spatial_center": self.spatial_center,
            "spatial_scale": self.spatial_scale,
            "t_min": self.t_min,
            "t_max": self.t_max,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(state, filepath)
        logger.info("Temporal SDF saved to %s", filepath)

    def load(self, filepath: str):
        """Load model from a file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self.model = TemporalIGRNetwork(
            in_dim=3, out_dim=1,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.is_fitted = checkpoint["is_fitted"]
        self.spatial_center = checkpoint["spatial_center"]
        self.spatial_scale = checkpoint["spatial_scale"]
        self.t_min = checkpoint["t_min"]
        self.t_max = checkpoint["t_max"]

        logger.info("Temporal SDF loaded from %s", filepath)

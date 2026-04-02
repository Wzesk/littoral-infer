"""
3D Temporal SDF — topology-aware shoreline interpolation and hindcasting.

Maps (x, y, t) -> SDF value using a neural implicit field. The zero
level-set at any time slice gives the shoreline for that day.

Key features:
- Spatial Eikonal constraint (|grad_xy phi| ~= 1) for valid distance fields
- Topology-aware decoding for periodic and non-periodic shorelines
- Endpoint anchoring for open coastlines
- Ensemble hindcasting with leave-one-out confidence calibration
- Convenience loaders from per-date 2D ShapeSDF checkpoints
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d

try:
    from skimage import measure
except ImportError:  # pragma: no cover - exercised indirectly in downstream integration
    measure = None

logger = logging.getLogger(__name__)


def _find_zero_contours(grid: np.ndarray) -> List[np.ndarray]:
    """Return zero-level contours with a skimage-first, matplotlib-second fallback."""
    if measure is not None:
        return measure.find_contours(grid, 0.0)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        axis = fig.add_subplot(111)
        cs = axis.contour(grid, levels=[0.0])
        contours: list[np.ndarray] = []
        if hasattr(cs, "collections"):
            for collection in cs.collections:
                for path in collection.get_paths():
                    verts = path.vertices
                    if len(verts) >= 2:
                        contours.append(np.asarray(verts, dtype=np.float32))
        else:
            for segment_group in getattr(cs, "allsegs", []):
                for verts in segment_group:
                    if len(verts) >= 2:
                        contours.append(np.asarray(verts, dtype=np.float32))
        return contours
    finally:
        plt.close(fig)


def _ensure_open(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    return pts


def _polyline_length(points: np.ndarray, periodic: bool) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    if periodic:
        pts = _ensure_open(pts)
        pts = np.vstack([pts, pts[0]])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _resample_polyline(points: np.ndarray, n_points: int, periodic: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n_points, axis=0)

    pts = _ensure_open(pts)
    if periodic:
        work = np.vstack([pts, pts[0]])
    else:
        work = pts

    seg = np.linalg.norm(np.diff(work, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 1e-8:
        return np.repeat(work[:1], n_points, axis=0)

    interp = interp1d(cum, work, axis=0, kind="linear")
    targets = np.linspace(0.0, cum[-1], n_points, endpoint=not periodic)
    out = interp(targets).astype(np.float32)
    if periodic and len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < 1e-6:
        out = out[:-1]
    return out


def _polygon_area(points: np.ndarray) -> float:
    pts = _ensure_open(points)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _canonicalize_periodic(points: np.ndarray) -> np.ndarray:
    pts = _ensure_open(points)
    if len(pts) < 3:
        return pts.astype(np.float32)
    if _polygon_area(pts) > 0:
        pts = pts[::-1]
    start_idx = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    return np.roll(pts, -start_idx, axis=0).astype(np.float32)


def _orient_open_to_anchors(points: np.ndarray, start_anchor: np.ndarray, end_anchor: np.ndarray) -> np.ndarray:
    pts = _ensure_open(points)
    if len(pts) < 2:
        return pts.astype(np.float32)
    direct = np.linalg.norm(pts[0] - start_anchor) + np.linalg.norm(pts[-1] - end_anchor)
    flipped = np.linalg.norm(pts[-1] - start_anchor) + np.linalg.norm(pts[0] - end_anchor)
    return pts.astype(np.float32) if direct <= flipped else pts[::-1].astype(np.float32)


def _nearest_shift_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pts = _ensure_open(points)
    ref = _ensure_open(reference)
    if len(pts) == 0 or len(ref) == 0:
        return pts.astype(np.float32)
    if len(pts) != len(ref):
        pts = _resample_polyline(pts, len(ref), periodic=True)
    distances = np.linalg.norm(pts - ref[0], axis=1)
    start_idx = int(np.argmin(distances))
    rolled = np.roll(pts, -start_idx, axis=0)
    if np.mean(np.linalg.norm(rolled - ref, axis=1)) > np.mean(np.linalg.norm(rolled[::-1] - ref, axis=1)):
        rolled = rolled[::-1]
    return rolled.astype(np.float32)


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    da = np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1)
    db = np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1)
    return float(0.5 * (da.mean() + db.mean()))


def _pointwise_spread(predictions: np.ndarray) -> np.ndarray:
    if len(predictions) == 0:
        return np.zeros(0, dtype=np.float32)
    mean = predictions.mean(axis=0)
    diffs = predictions - mean[None, :, :]
    return np.sqrt(np.sum(np.var(diffs, axis=0), axis=1)).astype(np.float32)


class TemporalIGRNetwork(nn.Module):
    """MLP mapping (x, y, t) -> SDF value."""

    def __init__(self, in_dim: int = 3, out_dim: int = 1, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Softplus(beta=100))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Softplus(beta=100))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShorelineEvolutionSDF:
    """Neural implicit (x, y, t) -> SDF for temporal shoreline modelling."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 6, device: str = "cpu"):
        self.device = device
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        self.model = TemporalIGRNetwork(3, 1, hidden_dim, num_layers).to(device)
        self.is_fitted = False

        self.spatial_center: Optional[np.ndarray] = None
        self.spatial_scale: Optional[float] = None
        self.t_min: Optional[float] = None
        self.t_max: Optional[float] = None

        self.periodic: Optional[bool] = None
        self.reference_boundary: Optional[np.ndarray] = None
        self.reference_start: Optional[np.ndarray] = None
        self.reference_end: Optional[np.ndarray] = None
        self.endpoint_mean: Optional[np.ndarray] = None
        self.endpoint_std: Optional[np.ndarray] = None
        self.training_days: Optional[np.ndarray] = None
        self.endpoint_anchor_weight: float = 0.25

    def _infer_periodic(self, boundary_points_per_day: List[np.ndarray]) -> bool:
        closed_votes = 0
        total = 0
        for pts in boundary_points_per_day:
            arr = np.asarray(pts, dtype=np.float32)
            if len(arr) < 3:
                continue
            total += 1
            perimeter = max(_polyline_length(arr, periodic=False), 1.0)
            if np.linalg.norm(arr[0] - arr[-1]) <= 0.03 * perimeter:
                closed_votes += 1
        return True if total == 0 else (closed_votes >= total / 2.0)

    def _prepare_boundary_sets(
        self,
        boundary_points_per_day: List[np.ndarray],
        periodic: Optional[bool] = None,
    ) -> List[np.ndarray]:
        use_periodic = self._infer_periodic(boundary_points_per_day) if periodic is None else bool(periodic)
        cleaned = [_ensure_open(np.asarray(pts, dtype=np.float32)) for pts in boundary_points_per_day]
        self.periodic = use_periodic

        if use_periodic:
            canonical = [_canonicalize_periodic(pts) for pts in cleaned]
            self.reference_boundary = _resample_polyline(canonical[0], 256, periodic=True)
            self.reference_start = self.reference_boundary[0].copy()
            self.reference_end = self.reference_boundary[-1].copy()
            self.endpoint_mean = None
            self.endpoint_std = None
            return canonical

        reference = cleaned[0].copy()
        aligned = []
        for pts in cleaned:
            aligned.append(_orient_open_to_anchors(pts, reference[0], reference[-1]))

        starts = np.stack([pts[0] for pts in aligned], axis=0)
        ends = np.stack([pts[-1] for pts in aligned], axis=0)
        self.endpoint_mean = np.stack([starts.mean(axis=0), ends.mean(axis=0)], axis=0).astype(np.float32)
        self.endpoint_std = np.stack([starts.std(axis=0), ends.std(axis=0)], axis=0).astype(np.float32)
        self.reference_boundary = _resample_polyline(aligned[0], 256, periodic=False)
        self.reference_start = self.endpoint_mean[0].copy()
        self.reference_end = self.endpoint_mean[1].copy()
        return aligned

    def _grid_contour_to_world(self, contour: np.ndarray, grid_res: int) -> np.ndarray:
        r, c = contour[:, 0], contour[:, 1]
        x_n = -1.1 + (c / max(grid_res - 1, 1)) * 2.2
        y_n = -1.1 + (r / max(grid_res - 1, 1)) * 2.2
        norm_pts = np.column_stack([x_n, y_n])
        return (norm_pts * self.spatial_scale + self.spatial_center).astype(np.float32)

    def _select_contour(self, contours: List[np.ndarray], grid_res: int) -> np.ndarray:
        if self.periodic:
            return max(contours, key=len)

        open_contours = []
        edge_dist = 2.0
        min_len = max(10, grid_res // 10)
        for contour in contours:
            if len(contour) < min_len:
                continue
            if np.linalg.norm(contour[0] - contour[-1]) <= 2.0:
                continue
            s, e = contour[0], contour[-1]
            s_ok = (s[0] <= edge_dist or s[0] >= grid_res - 1 - edge_dist or
                    s[1] <= edge_dist or s[1] >= grid_res - 1 - edge_dist)
            e_ok = (e[0] <= edge_dist or e[0] >= grid_res - 1 - edge_dist or
                    e[1] <= edge_dist or e[1] >= grid_res - 1 - edge_dist)
            if s_ok and e_ok:
                open_contours.append(contour)

        candidates = open_contours if open_contours else contours
        if self.reference_start is None or self.reference_end is None:
            return max(candidates, key=len)

        best = None
        best_score = None
        for contour in candidates:
            world = self._grid_contour_to_world(contour, grid_res)
            world = _orient_open_to_anchors(world, self.reference_start, self.reference_end)
            endpoint_error = (
                np.linalg.norm(world[0] - self.reference_start)
                + np.linalg.norm(world[-1] - self.reference_end)
            )
            score = (endpoint_error, -len(contour))
            if best_score is None or score < best_score:
                best_score = score
                best = contour
        return best if best is not None else max(candidates, key=len)

    def _postprocess_world_contour(self, world_pts: np.ndarray, n_points: int) -> np.ndarray:
        if self.periodic:
            periodic_pts = _canonicalize_periodic(world_pts)
            periodic_pts = _resample_polyline(periodic_pts, n_points, periodic=True)
            if self.reference_boundary is not None:
                ref = _resample_polyline(self.reference_boundary, len(periodic_pts), periodic=True)
                periodic_pts = _nearest_shift_to_reference(periodic_pts, ref)
            return periodic_pts

        open_pts = _orient_open_to_anchors(world_pts, self.reference_start, self.reference_end)
        open_pts = _resample_polyline(open_pts, n_points, periodic=False)
        if self.endpoint_mean is not None and len(open_pts) >= 2:
            open_pts[0] = (
                (1.0 - self.endpoint_anchor_weight) * open_pts[0]
                + self.endpoint_anchor_weight * self.endpoint_mean[0]
            )
            open_pts[-1] = (
                (1.0 - self.endpoint_anchor_weight) * open_pts[-1]
                + self.endpoint_anchor_weight * self.endpoint_mean[1]
            )
        return open_pts.astype(np.float32)

    def fit(
        self,
        boundary_points_per_day: List[np.ndarray],
        days: List[float],
        *,
        iterations: int = 3000,
        lr: float = 1e-3,
        lambda_eik: float = 0.1,
        n_rand_ratio: int = 2,
        periodic: Optional[bool] = None,
    ):
        assert len(boundary_points_per_day) == len(days)

        prepared_sets = self._prepare_boundary_sets(boundary_points_per_day, periodic=periodic)
        all_pts = np.concatenate(prepared_sets, axis=0)
        self.spatial_center = all_pts.mean(axis=0).astype(np.float32)
        self.spatial_scale = float(np.max(np.linalg.norm(all_pts - self.spatial_center, axis=1))) * 1.1
        self.training_days = np.asarray(days, dtype=np.float32)

        self.t_min = float(min(days))
        self.t_max = float(max(days))
        t_range = max(self.t_max - self.t_min, 1.0)

        manifold_list = []
        for pts, day in zip(prepared_sets, days):
            norm_xy = (pts - self.spatial_center) / self.spatial_scale
            t_col = np.full((len(pts), 1), (day - self.t_min) / t_range, dtype=np.float32)
            manifold_list.append(np.hstack([norm_xy, t_col]))

        manifold = torch.tensor(
            np.concatenate(manifold_list, axis=0),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        n_manifold = manifold.shape[0]
        logger.info("Training 3D SDF on %d manifold points over %d days...", n_manifold, len(days))

        for it in range(iterations):
            optimizer.zero_grad()
            pred = self.model(manifold)
            loss_manifold = pred.abs().mean()

            n_rand = n_manifold * n_rand_ratio
            xy_rand = (torch.rand(n_rand, 2, device=self.device) * 2 - 1)
            t_rand = torch.rand(n_rand, 1, device=self.device)
            rand_pts = torch.cat([xy_rand, t_rand], dim=1).requires_grad_(True)

            all_pts_t = torch.cat([manifold, rand_pts], dim=0)
            y = self.model(all_pts_t)

            grads = torch.autograd.grad(
                outputs=y,
                inputs=all_pts_t,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
            )[0]

            spatial_grad = grads[:, :2]
            grad_norm = spatial_grad.norm(2, dim=1)
            loss_eikonal = ((grad_norm - 1) ** 2).mean()

            loss = loss_manifold + lambda_eik * loss_eikonal
            loss.backward()
            optimizer.step()

            if (it + 1) % 500 == 0:
                logger.info(
                    "  iter %d/%d  manifold=%.5f  eikonal=%.5f",
                    it + 1,
                    iterations,
                    loss_manifold.item(),
                    loss_eikonal.item(),
                )

        self.is_fitted = True
        logger.info("3D SDF training complete.")

    def fit_from_sdf_folder(
        self,
        sdf_folder: str,
        days: Optional[List[float]] = None,
        *,
        n_boundary_points: int = 300,
        periodic: Optional[bool] = None,
        **fit_kwargs,
    ):
        import sys

        refine_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "littoral_refine"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "littoral_refine"),
        ]
        for path in refine_candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        from sdf_shoreline import ShapeSDF

        pth_files = sorted(glob.glob(os.path.join(sdf_folder, "*.pth")))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {sdf_folder}")

        if days is None:
            days = list(range(len(pth_files)))
        assert len(days) == len(pth_files), f"days ({len(days)}) must match pth files ({len(pth_files)})"

        boundary_sets: List[np.ndarray] = []
        inferred_periodic = periodic
        for path in pth_files:
            sdf = ShapeSDF(device=self.device)
            sdf.load(path)
            if inferred_periodic is None:
                inferred_periodic = bool(sdf.periodic)
            elif bool(sdf.periodic) != bool(inferred_periodic):
                raise ValueError("Mixed periodic/non-periodic ShapeSDF checkpoints in one temporal fit.")
            boundary_sets.append(sdf.get_boundary_points(n_boundary_points))

        self.fit(boundary_sets, days, periodic=inferred_periodic, **fit_kwargs)

    def evaluate_sdf_grid(self, day: float, grid_res: int = 256) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        t_range = max(self.t_max - self.t_min, 1.0)
        t_norm = (day - self.t_min) / t_range

        x = np.linspace(-1.1, 1.1, grid_res)
        y = np.linspace(-1.1, 1.1, grid_res)
        X, Y = np.meshgrid(x, y)
        grid_xy = np.column_stack([X.ravel(), Y.ravel()])
        t_col = np.full((grid_xy.shape[0], 1), t_norm, dtype=np.float32)
        grid_xyt = np.hstack([grid_xy, t_col])

        with torch.no_grad():
            tensor_in = torch.tensor(grid_xyt, dtype=torch.float32, device=self.device)
            return self.model(tensor_in).cpu().numpy().reshape(grid_res, grid_res)

    def get_shoreline_at_day(self, day: float, *, n_points: int = 200, grid_res: int = 256) -> np.ndarray:
        Z = self.evaluate_sdf_grid(day, grid_res=grid_res)
        contours = _find_zero_contours(Z)
        if not contours:
            raise ValueError(f"No zero level-set at day {day}")

        contour = self._select_contour(contours, grid_res)
        world_pts = self._grid_contour_to_world(contour, grid_res)
        return self._postprocess_world_contour(world_pts, n_points=n_points)

    def predict_shoreline(self, future_day: float, n_points: int = 200) -> np.ndarray:
        return self.get_shoreline_at_day(future_day, n_points=n_points)

    def save(self, filepath: str):
        state = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "is_fitted": self.is_fitted,
            "spatial_center": self.spatial_center,
            "spatial_scale": self.spatial_scale,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "periodic": self.periodic,
            "reference_boundary": self.reference_boundary,
            "reference_start": self.reference_start,
            "reference_end": self.reference_end,
            "endpoint_mean": self.endpoint_mean,
            "endpoint_std": self.endpoint_std,
            "training_days": self.training_days,
            "endpoint_anchor_weight": self.endpoint_anchor_weight,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(state, filepath)
        logger.info("Temporal SDF saved to %s", filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.model = TemporalIGRNetwork(
            in_dim=3,
            out_dim=1,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.is_fitted = checkpoint["is_fitted"]
        self.spatial_center = checkpoint["spatial_center"]
        self.spatial_scale = checkpoint["spatial_scale"]
        self.t_min = checkpoint["t_min"]
        self.t_max = checkpoint["t_max"]
        self.periodic = checkpoint.get("periodic")
        self.reference_boundary = checkpoint.get("reference_boundary")
        self.reference_start = checkpoint.get("reference_start")
        self.reference_end = checkpoint.get("reference_end")
        self.endpoint_mean = checkpoint.get("endpoint_mean")
        self.endpoint_std = checkpoint.get("endpoint_std")
        self.training_days = checkpoint.get("training_days")
        self.endpoint_anchor_weight = float(checkpoint.get("endpoint_anchor_weight", 0.25))
        logger.info("Temporal SDF loaded from %s", filepath)


class TemporalSDFEnsemble:
    """Bootstrap ensemble of topology-aware temporal SDF models."""

    def __init__(
        self,
        n_members: int = 5,
        *,
        model_kwargs: Optional[Dict] = None,
        bootstrap_point_ratio: float = 0.75,
        seed: int = 7,
    ):
        self.n_members = int(n_members)
        self.model_kwargs = dict(model_kwargs or {})
        self.bootstrap_point_ratio = float(bootstrap_point_ratio)
        self.seed = int(seed)
        self.members: List[ShorelineEvolutionSDF] = []
        self.periodic: Optional[bool] = None
        self.reference_boundary: Optional[np.ndarray] = None
        self.reference_start: Optional[np.ndarray] = None
        self.reference_end: Optional[np.ndarray] = None
        self.calibration: Dict[str, float] = {}
        self.training_days: Optional[np.ndarray] = None

    def _subsample_points(self, pts: np.ndarray, rng: np.random.Generator, periodic: bool) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float32)
        count = max(16, int(round(len(arr) * self.bootstrap_point_ratio)))
        count = min(count, len(arr))
        if count >= len(arr):
            return arr.copy()
        indices = np.sort(rng.choice(len(arr), size=count, replace=False))
        sampled = arr[indices]
        return _ensure_open(sampled if not periodic else sampled)

    def _align_observation(self, points: np.ndarray, n_points: int) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        if self.periodic:
            pts = _canonicalize_periodic(pts)
            pts = _resample_polyline(pts, n_points, periodic=True)
            if self.reference_boundary is not None:
                ref = _resample_polyline(self.reference_boundary, n_points, periodic=True)
                pts = _nearest_shift_to_reference(pts, ref)
            return pts

        pts = _orient_open_to_anchors(pts, self.reference_start, self.reference_end)
        pts = _resample_polyline(pts, n_points, periodic=False)
        return pts

    def fit(
        self,
        boundary_points_per_day: List[np.ndarray],
        days: List[float],
        *,
        periodic: Optional[bool] = None,
        fit_kwargs: Optional[Dict] = None,
    ) -> "TemporalSDFEnsemble":
        fit_kwargs = dict(fit_kwargs or {})
        rng = np.random.default_rng(self.seed)
        self.training_days = np.asarray(days, dtype=np.float32)
        self.members = []

        # Infer periodicity from an initial template fit if not provided.
        template = ShorelineEvolutionSDF(**self.model_kwargs)
        template.fit(boundary_points_per_day, days, periodic=periodic, **fit_kwargs)
        self.periodic = template.periodic
        self.reference_boundary = template.reference_boundary
        self.reference_start = template.reference_start
        self.reference_end = template.reference_end
        self.members.append(template)

        for _ in range(1, self.n_members):
            sampled_sets = [
                self._subsample_points(pts, rng, self.periodic)
                for pts in boundary_points_per_day
            ]
            member = ShorelineEvolutionSDF(**self.model_kwargs)
            member.fit(sampled_sets, days, periodic=self.periodic, **fit_kwargs)
            self.members.append(member)

        return self

    def predict(self, day: float, *, n_points: int = 200, grid_res: int = 256) -> Dict[str, np.ndarray]:
        if not self.members:
            raise RuntimeError("Ensemble not fitted.")

        member_predictions = []
        for member in self.members:
            pred = member.get_shoreline_at_day(day, n_points=n_points, grid_res=grid_res)
            if self.periodic and self.reference_boundary is not None:
                pred = _nearest_shift_to_reference(pred, _resample_polyline(self.reference_boundary, n_points, periodic=True))
            elif not self.periodic:
                pred = _orient_open_to_anchors(pred, self.reference_start, self.reference_end)
            member_predictions.append(pred)

        stack = np.stack(member_predictions, axis=0).astype(np.float32)
        mean_shoreline = stack.mean(axis=0)
        point_spread = _pointwise_spread(stack)
        global_spread = float(point_spread.mean()) if len(point_spread) else 0.0
        calibration_scale = float(self.calibration.get("spread_to_error_scale", 1.0))
        predicted_error = calibration_scale * global_spread
        reference_scale = max(_polyline_length(mean_shoreline, periodic=bool(self.periodic)) / max(len(mean_shoreline), 1), 1.0)
        confidence_score = float(np.exp(-predicted_error / reference_scale))

        return {
            "shoreline": mean_shoreline.astype(np.float32),
            "member_shorelines": stack,
            "pointwise_spread": point_spread,
            "global_spread": np.array(global_spread, dtype=np.float32),
            "predicted_error": np.array(predicted_error, dtype=np.float32),
            "confidence_score": np.array(confidence_score, dtype=np.float32),
        }

    def calibrate_leave_one_out(
        self,
        boundary_points_per_day: List[np.ndarray],
        days: List[float],
        *,
        n_points: int = 200,
        grid_res: int = 256,
        fit_kwargs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        if len(days) < 3:
            self.calibration = {"spread_to_error_scale": 1.0, "loo_samples": float(len(days))}
            return self.calibration

        fit_kwargs = dict(fit_kwargs or {})
        ratios = []
        errors = []
        spreads = []

        for idx, day in enumerate(days):
            train_pts = [pts for j, pts in enumerate(boundary_points_per_day) if j != idx]
            train_days = [d for j, d in enumerate(days) if j != idx]
            temp = TemporalSDFEnsemble(
                n_members=self.n_members,
                model_kwargs=self.model_kwargs,
                bootstrap_point_ratio=self.bootstrap_point_ratio,
                seed=self.seed + idx + 1,
            )
            temp.fit(train_pts, train_days, periodic=self.periodic, fit_kwargs=fit_kwargs)
            pred = temp.predict(day, n_points=n_points, grid_res=grid_res)
            observed = temp._align_observation(boundary_points_per_day[idx], n_points=n_points)
            error = np.linalg.norm(pred["shoreline"] - observed, axis=1)
            mean_error = float(error.mean())
            mean_spread = float(pred["pointwise_spread"].mean())
            errors.append(mean_error)
            spreads.append(mean_spread)
            ratios.append(mean_error / max(mean_spread, 1e-6))

        spread_to_error_scale = float(np.median(np.clip(ratios, 0.5, 10.0))) if ratios else 1.0
        self.calibration = {
            "spread_to_error_scale": spread_to_error_scale,
            "loo_mean_error": float(np.mean(errors)) if errors else 0.0,
            "loo_mean_spread": float(np.mean(spreads)) if spreads else 0.0,
            "loo_samples": float(len(errors)),
        }
        return self.calibration

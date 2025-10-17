"""Pure PyTorch fallbacks for the PointNet++ custom ops.

These implementations mirror the CUDA extension defined under
`Pose_Estimation_Model/model/pointnet2/_ext_src`. They are significantly
slower than the fused kernels, but unblock environments where the custom
extension cannot be compiled. All functions expect PyTorch tensors that
match the shapes used by the original ops.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _assert_has_shape(tensor: Tensor, dims: int, name: str) -> None:
    if tensor.dim() != dims:
        raise ValueError(f"Expected {name} to have {dims} dims, got {tensor.dim()}")


@torch.no_grad()
def furthest_point_sampling(xyz: Tensor, npoint: int) -> Tensor:
    """Iterative furthest point sampling without the custom CUDA kernel."""
    _assert_has_shape(xyz, 3, "xyz")
    if xyz.size(-1) != 3:
        raise ValueError("Expected xyz to have shape (B, N, 3)")
    if npoint <= 0 or npoint > xyz.size(1):
        raise ValueError("npoint must be in (0, N]")

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        closer = dist < distance
        distance[closer] = dist[closer]
        farthest = torch.max(distance, dim=1)[1]

    return centroids


def gather_points(points: Tensor, idx: Tensor) -> Tensor:
    """Gather features using the provided indices."""
    _assert_has_shape(points, 3, "points")
    _assert_has_shape(idx, 2, "idx")
    batch, channels, _ = points.shape
    if idx.size(0) != batch:
        raise ValueError("Batch size mismatch between points and idx")

    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1)
    return torch.gather(points, 2, idx_expanded)


def gather_points_grad(grad_out: Tensor, idx: Tensor, n: int) -> Tensor:
    """Scatter gradients back to the original feature set."""
    _assert_has_shape(grad_out, 3, "grad_out")
    _assert_has_shape(idx, 2, "idx")
    if grad_out.size(0) != idx.size(0):
        raise ValueError("Batch size mismatch between grad_out and idx")
    if n <= 0:
        raise ValueError("Expected n > 0")

    batch, channels, _ = grad_out.shape
    grad_points = torch.zeros(batch, channels, n, device=grad_out.device, dtype=grad_out.dtype)
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1)
    grad_points.scatter_add_(2, idx_expanded, grad_out)
    return grad_points


@torch.no_grad()
def ball_query(xyz: Tensor, new_xyz: Tensor, radius: float, nsample: int) -> Tensor:
    """Find up to `nsample` neighbors within `radius` of each query point."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    _assert_has_shape(xyz, 3, "xyz")
    _assert_has_shape(new_xyz, 3, "new_xyz")
    if xyz.size(-1) != 3 or new_xyz.size(-1) != 3:
        raise ValueError("Input tensors must end with dimension 3")
    if xyz.size(0) != new_xyz.size(0):
        raise ValueError("Batch size mismatch between xyz and new_xyz")

    batch_size, num_points, _ = xyz.shape
    _, num_queries, _ = new_xyz.shape
    idx = torch.zeros(batch_size, num_queries, nsample, dtype=torch.long, device=xyz.device)
    radius2 = radius * radius

    for b in range(batch_size):
        source = xyz[b]
        queries = new_xyz[b]
        dist2 = torch.sum((queries.unsqueeze(1) - source.unsqueeze(0)) ** 2, dim=-1)
        for j in range(num_queries):
            mask = dist2[j] < radius2
            neighbors = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if neighbors.numel() == 0:
                continue
            if neighbors.numel() >= nsample:
                chosen = neighbors[:nsample]
            else:
                pad = neighbors[0].repeat(nsample - neighbors.numel())
                chosen = torch.cat([neighbors, pad], dim=0)
            idx[b, j] = chosen

    return idx


@torch.no_grad()
def three_nn(unknown: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
    """Return squared distances and indices for the three nearest neighbors."""
    _assert_has_shape(unknown, 3, "unknown")
    _assert_has_shape(known, 3, "known")
    if unknown.size(-1) != 3 or known.size(-1) != 3:
        raise ValueError("Point tensors must end with dimension 3")
    if unknown.size(0) != known.size(0):
        raise ValueError("Batch size mismatch between unknown and known")
    if known.size(1) < 3:
        raise ValueError("Need at least three known points")

    dist = torch.cdist(unknown, known, p=2) ** 2
    dist2, idx = torch.topk(dist, k=3, dim=-1, largest=False, sorted=True)
    return dist2, idx


def three_interpolate(points: Tensor, idx: Tensor, weight: Tensor) -> Tensor:
    """Weighted linear interpolation over three neighbors."""
    _assert_has_shape(points, 3, "points")
    _assert_has_shape(idx, 3, "idx")
    _assert_has_shape(weight, 3, "weight")
    if idx.size(-1) != 3 or weight.size(-1) != 3:
        raise ValueError("idx and weight must have last dimension 3")
    if points.size(0) != idx.size(0) or idx.shape[:2] != weight.shape[:2]:
        raise ValueError("Batch or spatial shape mismatch")

    batch, channels, _ = points.shape
    _, num_queries, _ = idx.shape
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    gathered = torch.gather(points.unsqueeze(2).expand(-1, -1, num_queries, -1), 3, idx_expanded)
    weights = weight.unsqueeze(1)
    return torch.sum(gathered * weights, dim=-1)


def three_interpolate_grad(grad_out: Tensor, idx: Tensor, weight: Tensor, m: int) -> Tensor:
    """Back-propagate gradients for three-point interpolation."""
    _assert_has_shape(grad_out, 3, "grad_out")
    _assert_has_shape(idx, 3, "idx")
    _assert_has_shape(weight, 3, "weight")
    if m <= 0:
        raise ValueError("m must be positive")
    if idx.size(-1) != 3 or weight.size(-1) != 3:
        raise ValueError("idx and weight must have last dimension 3")
    if grad_out.shape[:2] != idx.shape[:2] or grad_out.shape[:2] != weight.shape[:2]:
        raise ValueError("Shape mismatch between grad_out, idx, and weight")

    batch, channels, _ = grad_out.shape
    grad_points = torch.zeros(batch, channels, m, device=grad_out.device, dtype=grad_out.dtype)
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    weights = weight.unsqueeze(1)
    contrib = grad_out.unsqueeze(-1) * weights
    grad_points.scatter_add_(2, idx_expanded.reshape(batch, channels, -1), contrib.reshape(batch, channels, -1))
    return grad_points


def group_points(points: Tensor, idx: Tensor) -> Tensor:
    """Gather groups of points for local neighborhoods."""
    _assert_has_shape(points, 3, "points")
    _assert_has_shape(idx, 3, "idx")
    batch, channels, num_points = points.shape
    if idx.size(0) != batch:
        raise ValueError("Batch size mismatch between points and idx")
    npoint, nsample = idx.size(1), idx.size(2)

    points_expanded = points.unsqueeze(2).expand(-1, -1, npoint, -1)
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    return torch.gather(points_expanded, 3, idx_expanded)


def group_points_grad(grad_out: Tensor, idx: Tensor, n: int) -> Tensor:
    """Scatter neighborhood gradients back to the original features."""
    _assert_has_shape(grad_out, 4, "grad_out")
    _assert_has_shape(idx, 3, "idx")
    if grad_out.size(0) != idx.size(0) or grad_out.size(2) != idx.size(1) or grad_out.size(3) != idx.size(2):
        raise ValueError("Shape mismatch between grad_out and idx")
    if n <= 0:
        raise ValueError("n must be positive")

    batch, channels, _, _ = grad_out.shape
    grad_points = torch.zeros(batch, channels, n, device=grad_out.device, dtype=grad_out.dtype)
    idx_expanded = idx.unsqueeze(1).expand(-1, channels, -1, -1)
    grad_points.scatter_add_(
        2,
        idx_expanded.reshape(batch, channels, -1),
        grad_out.reshape(batch, channels, -1),
    )
    return grad_points


__all__ = [
    "furthest_point_sampling",
    "gather_points",
    "gather_points_grad",
    "ball_query",
    "three_nn",
    "three_interpolate",
    "three_interpolate_grad",
    "group_points",
    "group_points_grad",
]

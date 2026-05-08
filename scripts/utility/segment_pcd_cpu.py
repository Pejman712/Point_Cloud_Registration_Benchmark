#!/usr/bin/env python3
"""
CPU-only geometric segmentation for a PCD file.

This does NOT use SFPNet.
It segments the cloud into:
  - dominant plane, often floor/wall/ground
  - Euclidean clusters for remaining points

Usage:
    python3 segment_pcd_cpu.py \
      --pcd input.pcd \
      --out output_segmented_cpu.pcd \
      --labels-out output_labels_cpu.npy
"""

import argparse
import numpy as np
import open3d as o3d


def make_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    colors = rng.integers(30, 255, size=(max(n, 1), 3), dtype=np.uint8)
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--out", default="output_segmented_cpu.pcd")
    parser.add_argument("--labels-out", default="output_labels_cpu.npy")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--plane-threshold", type=float, default=0.08)
    parser.add_argument("--cluster-eps", type=float, default=0.35)
    parser.add_argument("--min-points", type=int, default=20)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd)
    if pcd.is_empty():
        raise RuntimeError(f"Could not read point cloud: {args.pcd}")

    original_points = np.asarray(pcd.points)
    print(f"Loaded {len(original_points)} points")

    # Downsample for faster CPU segmentation.
    pcd_ds = pcd.voxel_down_sample(args.voxel_size)
    points_ds = np.asarray(pcd_ds.points)
    print(f"Downsampled to {len(points_ds)} points")

    # Segment dominant plane.
    plane_model, plane_indices = pcd_ds.segment_plane(
        distance_threshold=args.plane_threshold,
        ransac_n=3,
        num_iterations=1000,
    )

    labels_ds = np.full(len(points_ds), -1, dtype=np.int32)
    labels_ds[plane_indices] = 0

    non_plane = pcd_ds.select_by_index(plane_indices, invert=True)
    non_plane_indices = np.setdiff1d(np.arange(len(points_ds)), np.array(plane_indices))

    cluster_labels = np.array(
        non_plane.cluster_dbscan(
            eps=args.cluster_eps,
            min_points=args.min_points,
            print_progress=True,
        ),
        dtype=np.int32,
    )

    valid = cluster_labels >= 0
    labels_ds[non_plane_indices[valid]] = cluster_labels[valid] + 1

    # Map downsampled labels back to original points using nearest neighbor.
    kdtree = o3d.geometry.KDTreeFlann(pcd_ds)
    labels = np.empty(len(original_points), dtype=np.int32)

    for i, pt in enumerate(original_points):
        _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
        labels[i] = labels_ds[idx[0]]

    max_label = int(labels.max())
    palette = make_palette(max_label + 2)

    colors = np.zeros((len(labels), 3), dtype=np.float64)

    # Unknown/noise = dark gray
    colors[labels < 0] = np.array([0.15, 0.15, 0.15])

    # Plane = light gray
    colors[labels == 0] = np.array([0.75, 0.75, 0.75])

    # Clusters
    for lab in range(1, max_label + 1):
        colors[labels == lab] = palette[lab] / 255.0

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(original_points)
    out.colors = o3d.utility.Vector3dVector(colors)

    np.save(args.labels_out, labels)
    o3d.io.write_point_cloud(args.out, out)

    print(f"Unique labels: {np.unique(labels)}")
    print(f"Wrote labels: {args.labels_out}")
    print(f"Wrote colored PCD: {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import urllib.error
import urllib.request

import numpy as np
import open3d as o3d
import open3d.ml as _ml3d


S3DIS_CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

S3DIS_COLORS = np.array(
    [
        [0.65, 0.65, 0.65],  # ceiling
        [0.35, 0.35, 0.35],  # floor
        [0.80, 0.40, 0.40],  # wall
        [0.60, 0.30, 0.30],  # beam
        [0.60, 0.60, 0.20],  # column
        [0.20, 0.60, 0.80],  # window
        [0.80, 0.60, 0.20],  # door
        [0.40, 0.80, 0.40],  # table
        [0.20, 0.80, 0.20],  # chair
        [0.80, 0.20, 0.80],  # sofa
        [0.50, 0.30, 0.80],  # bookcase
        [0.20, 0.30, 0.80],  # board
        [0.90, 0.90, 0.20],  # clutter
    ],
    dtype=np.float32,
)

GENERALIZE_CLASSES = ["floor", "wall", "ceiling"]

KNOWN_MD5 = {
    "randlanet_s3dis_202201071330utc.pth": None,
    "randlanet_s3dis_202010091238.pth": "5f993ef4a52065e4f882764568e6f378",
    "randlanet_s3dis_area5_202010091333utc.pth": "44e45c8244a70e899065d1c1c4719b73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment a single PCD file using Open3D-ML RandLA-Net on CPU, then optionally generalize floor/wall/ceiling."
    )

    parser.add_argument("--input", required=True, help="Input .pcd file")
    parser.add_argument("--output", "--out", required=True, help="Output segmented .pcd file")

    parser.add_argument(
        "--open3d_ml_root",
        required=True,
        help="Path to cloned Open3D-ML repo, e.g. ../../lib/Open3D-ML",
    )

    parser.add_argument(
        "--cfg_file",
        default=None,
        help="Open3D-ML config file. Default: <open3d_ml_root>/ml3d/configs/randlanet_s3dis.yml",
    )

    parser.add_argument(
        "--ckpt",
        "--ckpt_path",
        default="./weights/randlanet_s3dis_202201071330utc.pth",
        help="Path to RandLA-Net S3DIS .pth checkpoint.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel downsampling size before inference. Use 0 to disable.",
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=45056,
        help="RandLA-Net number of sampled points. Lower is lighter on CPU.",
    )

    parser.add_argument(
        "--no_auto_download",
        action="store_true",
        help="Disable automatic checkpoint download.",
    )

    parser.add_argument(
        "--feature_mode",
        choices=["auto", "zeros", "geometry"],
        default="auto",
        help=(
            "Feature fallback when input PCD has no RGB. "
            "auto=RGB if available else geometry; zeros=zero RGB; geometry=height/range/ones."
        ),
    )

    parser.add_argument(
        "--generalize",
        action="store_true",
        help="Create an additional generalized PCD for floor, wall, and ceiling.",
    )

    parser.add_argument(
        "--generalize_distance",
        type=float,
        default=0.08,
        help="RANSAC plane inlier distance threshold for generalization.",
    )

    parser.add_argument(
        "--generalize_voxel_size",
        type=float,
        default=0.10,
        help="Voxel size used after projecting generalized planar points. Use 0 to disable.",
    )

    parser.add_argument(
        "--generate_random_points",
        action="store_true",
        help="Generate additional random synthetic points on the detected floor/wall/ceiling planes.",
    )

    parser.add_argument(
        "--random_points_per_class",
        type=int,
        default=5000,
        help="Number of random synthetic points to add per generalized class.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=7,
        help="Random seed used for synthetic point generation.",
    )

    parser.add_argument(
        "--max_planes_per_class",
        type=int,
        default=10,
        help="Maximum number of RANSAC planes to extract for each generalized class.",
    )

    parser.add_argument(
        "--min_plane_points",
        type=int,
        default=300,
        help="Minimum number of inlier points required to accept a detected plane.",
    )

    parser.add_argument(
        "--keep_generalized_only",
        action="store_true",
        help=(
            "When generalizing, save only floor/wall/ceiling generalized points. "
            "Without this flag, the generalized output also keeps non-generalized classes."
        ),
    )

    parser.add_argument(
        "--show_before_after",
        action="store_true",
        help="Show simple before and after point clouds. No legend, no class text, no advanced UI.",
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Open3D visualization point size.",
    )

    parser.add_argument(
        "--axis_size",
        type=float,
        default=1.0,
        help="Coordinate-frame axis size in Open3D visualization. Use 0 to hide.",
    )

    return parser.parse_args()


def md5sum(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_md5_if_known(path):
    filename = os.path.basename(path)
    expected = KNOWN_MD5.get(filename)

    if filename not in KNOWN_MD5:
        print(f"No MD5 entry for {filename}; skipping checksum verification.")
        return

    if expected is None:
        print(f"No fixed MD5 stored for {filename}; skipping checksum verification.")
        return

    actual = md5sum(path)

    if actual != expected:
        raise RuntimeError(
            "Checkpoint MD5 mismatch.\n"
            f"  File:     {path}\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            "Delete the file and download it again."
        )

    print(f"Checkpoint MD5 OK: {filename}")


def download_file(url, output_path, timeout=60):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Trying checkpoint URL:\n  {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                print(f"  Failed: HTTP {response.status}")
                return False

            tmp_path = output_path + ".tmp"

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)

            size = os.path.getsize(tmp_path)

            if size < 1024 * 1024:
                os.remove(tmp_path)
                print("  Failed: downloaded file is too small.")
                return False

            os.replace(tmp_path, output_path)
            print(f"Downloaded checkpoint:\n  {output_path}")
            return True

    except urllib.error.HTTPError as e:
        print(f"  Failed: HTTP {e.code}")
        return False

    except urllib.error.URLError as e:
        print(f"  Failed: URL error: {e}")
        return False

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def find_checkpoint_url_from_model_zoo(open3d_ml_root, filename):
    model_zoo_path = os.path.join(open3d_ml_root, "model_zoo.md")

    if not os.path.isfile(model_zoo_path):
        return None

    with open(model_zoo_path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"\((https?://[^)]*" + re.escape(filename) + r"[^)]*)\)"
    match = re.search(pattern, text)

    if match:
        return match.group(1)

    pattern = r"https?://\S*" + re.escape(filename)
    match = re.search(pattern, text)

    if match:
        return match.group(0).rstrip(")")

    return None


def auto_download_checkpoint(open3d_ml_root, ckpt_path):
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 1024 * 1024:
        print(f"Checkpoint already exists:\n  {ckpt_path}")
        return

    filename = os.path.basename(ckpt_path)

    urls = []

    url_from_model_zoo = find_checkpoint_url_from_model_zoo(open3d_ml_root, filename)

    if url_from_model_zoo:
        urls.append(url_from_model_zoo)

    urls.append(
        "https://storage.googleapis.com/open3d-releases/model-zoo/"
        "randlanet_s3dis_202201071330utc.pth"
    )

    urls.extend(
        [
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_202010091238.pth",
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_area5_202010091333utc.pth",
        ]
    )

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    attempted = []

    for url in urls:
        attempted.append(url)

        if download_file(url, ckpt_path):
            return

    raise FileNotFoundError(
        "Could not automatically download checkpoint.\n\n"
        "Tried:\n"
        + "\n".join(f"  {u}" for u in attempted)
        + "\n\nManual download command:\n"
        "  mkdir -p weights\n"
        "  wget -O ./weights/randlanet_s3dis_202201071330utc.pth "
        "https://storage.googleapis.com/open3d-releases/model-zoo/"
        "randlanet_s3dis_202201071330utc.pth\n"
    )


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise RuntimeError(f"Could not read point cloud or file is empty: {path}")

    points = np.asarray(pcd.points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError("Input point cloud must contain XYZ points.")

    return pcd


def make_features(pcd, points, feature_mode):
    """
    RandLA-Net S3DIS expects 3 feature channels in addition to XYZ.
    In S3DIS, these are RGB values. For LiDAR-only PCDs, this script
    synthesizes 3 feature channels.
    """

    if pcd.has_colors() and feature_mode == "auto":
        print("Using RGB colors from PCD as features.")

        feat = np.asarray(pcd.colors).astype(np.float32)

        if feat.max() > 1.0:
            feat = feat / 255.0

        return feat

    if feature_mode == "zeros":
        print("Input feature mode: zero RGB features.")
        return np.zeros((points.shape[0], 3), dtype=np.float32)

    print("Input PCD has no RGB colors. Using geometric pseudo-features.")

    z = points[:, 2:3]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

    xy_dist = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
    xy_dist = (xy_dist - xy_dist.min()) / (xy_dist.max() - xy_dist.min() + 1e-6)

    ones = np.ones_like(z_norm)

    feat = np.concatenate([z_norm, xy_dist, ones], axis=1).astype(np.float32)

    return feat


def print_class_counts(labels, title="Class counts"):
    print(f"\n{title}:")
    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if 0 <= label < len(S3DIS_CLASS_NAMES):
            class_name = S3DIS_CLASS_NAMES[label]
        else:
            class_name = "unknown"

        print(f"  {label:2d}  {class_name:10s}  {count}")


def save_label_npz(output_pcd_path, points, labels, scores=None):
    npz_path = os.path.splitext(output_pcd_path)[0] + "_labels.npz"

    if scores is None:
        scores = np.ones((points.shape[0],), dtype=np.float32)

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
        scores=scores.astype(np.float32),
        class_names=np.array(S3DIS_CLASS_NAMES),
    )

    print(f"Saved label data:\n  {npz_path}")


def create_colored_point_cloud(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    colors = S3DIS_COLORS[labels % len(S3DIS_COLORS)]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def save_segmented_pcd(output_path, points, labels):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    pcd = create_colored_point_cloud(points, labels)

    ok = o3d.io.write_point_cloud(
        output_path,
        pcd,
        write_ascii=False,
    )

    if not ok:
        raise RuntimeError(f"Failed to write point cloud: {output_path}")

    print(f"Saved point cloud:\n  {output_path}")


def show_point_cloud(pcd, window_name="Point Cloud", point_size=2.0, axis_size=1.0):
    """
    Simple Open3D viewer.
    No labels, no legend, no O3DVisualizer, no topic windows.
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    vis.add_geometry(pcd)

    if axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        vis.add_geometry(axis)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.asarray([0.02, 0.02, 0.02])

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.0, -1.0, 0.4])
    view_ctl.set_lookat(pcd.get_center())
    view_ctl.set_up([0.0, 0.0, 1.0])
    view_ctl.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


def show_original_and_generated(
    original_points,
    generated_points,
    point_size=2.0,
    axis_size=1.0,
):
    """
    Show one simple Open3D window:
      - original/downsampled input cloud in white
      - generated synthetic points in green
    """

    combined_pcd = o3d.geometry.PointCloud()

    if generated_points is None or generated_points.shape[0] == 0:
        all_points = original_points.astype(np.float64)
        all_colors = np.ones((original_points.shape[0], 3), dtype=np.float64)
    else:
        all_points = np.vstack([
            original_points.astype(np.float32),
            generated_points.astype(np.float32),
        ]).astype(np.float64)

        original_colors = np.ones((original_points.shape[0], 3), dtype=np.float64)
        generated_colors = np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            (generated_points.shape[0], 1),
        )
        all_colors = np.vstack([original_colors, generated_colors])

    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    print("Showing original cloud in white and generated points in green.")
    show_point_cloud(
        combined_pcd,
        window_name="Original Cloud White + Generated Points Green",
        point_size=point_size,
        axis_size=axis_size,
    )


# -------------------------------------------------------------------------
# Point generalization
# -------------------------------------------------------------------------


def extract_class_points(points, labels, class_name):
    if class_name not in S3DIS_CLASS_NAMES:
        raise ValueError(f"Unknown S3DIS class name: {class_name}")

    class_id = S3DIS_CLASS_NAMES.index(class_name)
    mask = labels == class_id

    return points[mask], mask, class_id


def fit_plane_ransac(
    points,
    distance_threshold=0.08,
    ransac_n=3,
    num_iterations=2000,
):
    if points.shape[0] < ransac_n:
        return None, []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    return plane_model, inliers


def project_points_to_plane(points, plane_model):
    a, b, c, d = plane_model

    normal = np.array([a, b, c], dtype=np.float32)
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-8:
        raise RuntimeError("Invalid plane normal.")

    normal = normal / normal_norm

    distances = points @ normal + d
    projected = points - distances[:, None] * normal[None, :]

    return projected.astype(np.float32)


def downsample_points_with_labels(points, labels, voxel_size):
    if points.shape[0] == 0:
        return points, labels

    if voxel_size <= 0:
        return points, labels

    pcd = create_colored_point_cloud(points, labels)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    down_points = np.asarray(pcd.points).astype(np.float32)

    if labels.shape[0] > 0:
        class_id = int(labels[0])
    else:
        class_id = 0

    down_labels = np.full(
        down_points.shape[0],
        class_id,
        dtype=np.int32,
    )

    return down_points, down_labels


def make_plane_basis(plane_model):
    """
    Create two orthonormal basis vectors that lie inside a plane.
    """

    a, b, c, _ = plane_model
    normal = np.array([a, b, c], dtype=np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    if abs(float(normal[2])) < 0.9:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    basis_u = np.cross(normal, reference)
    basis_u = basis_u / (np.linalg.norm(basis_u) + 1e-8)

    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / (np.linalg.norm(basis_v) + 1e-8)

    return basis_u.astype(np.float32), basis_v.astype(np.float32), normal.astype(np.float32)


def convex_hull_2d(points_2d):
    """
    Compute a 2D convex hull using Andrew's monotonic chain algorithm.
    Returns hull vertices in counter-clockwise order.
    """

    pts = np.unique(points_2d.astype(np.float64), axis=0)

    if pts.shape[0] <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def points_inside_convex_polygon(points_2d, polygon):
    """
    Check whether 2D points are inside a convex polygon.
    Boundary points are accepted.
    """

    if polygon.shape[0] < 3:
        return np.zeros(points_2d.shape[0], dtype=bool)

    inside = np.ones(points_2d.shape[0], dtype=bool)
    eps = 1e-9

    for i in range(polygon.shape[0]):
        a = polygon[i]
        b = polygon[(i + 1) % polygon.shape[0]]
        edge = b - a
        rel = points_2d - a[None, :]
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        inside &= cross >= -eps

    return inside


def generate_random_points_on_plane_region(
    plane_points,
    plane_model,
    num_points,
    random_seed=7,
):
    """
    Generate random synthetic points on the same bounded plane region as the detected points.

    This version samples inside the 2D convex hull of the detected planar points,
    not the full rectangular bounding box. This prevents generated points from
    spreading far outside the observed cloud boundary.
    """

    if num_points <= 0 or plane_points.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float32)

    basis_u, basis_v, normal = make_plane_basis(plane_model)

    origin = plane_points.mean(axis=0).astype(np.float32)
    relative = plane_points.astype(np.float32) - origin[None, :]

    coords_u = relative @ basis_u
    coords_v = relative @ basis_v
    coords_2d = np.column_stack([coords_u, coords_v]).astype(np.float64)

    hull = convex_hull_2d(coords_2d)

    if hull.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float32)

    u_min, v_min = hull.min(axis=0)
    u_max, v_max = hull.max(axis=0)

    rng = np.random.default_rng(random_seed)
    accepted = []
    remaining = int(num_points)
    max_rounds = 100

    for _ in range(max_rounds):
        if remaining <= 0:
            break

        # Oversample because rejection removes candidates outside the hull.
        batch_size = max(remaining * 3, 1000)
        sample_u = rng.uniform(u_min, u_max, size=batch_size)
        sample_v = rng.uniform(v_min, v_max, size=batch_size)
        candidates = np.column_stack([sample_u, sample_v])

        mask = points_inside_convex_polygon(candidates, hull)
        inside_candidates = candidates[mask]

        if inside_candidates.shape[0] == 0:
            continue

        take = min(remaining, inside_candidates.shape[0])
        accepted.append(inside_candidates[:take])
        remaining -= take

    if not accepted:
        return np.empty((0, 3), dtype=np.float32)

    sampled_2d = np.vstack(accepted).astype(np.float32)

    sampled_points = (
        origin[None, :]
        + sampled_2d[:, 0:1] * basis_u[None, :]
        + sampled_2d[:, 1:2] * basis_v[None, :]
    )

    sampled_points = project_points_to_plane(sampled_points, plane_model)

    return sampled_points.astype(np.float32)


def generalize_planar_class(
    points,
    labels,
    class_name,
    distance_threshold=0.08,
    voxel_size=0.10,
    generate_random_points=False,
    random_points_per_class=5000,
    random_seed=7,
    max_planes_per_class=10,
    min_plane_points=300,
):
    """
    Extract multiple planar patches for one semantic class.

    Instead of fitting only one dominant plane, this repeatedly:
      1. fits a RANSAC plane,
      2. removes its inliers,
      3. fits another plane to the remaining points.

    This allows around max_planes_per_class planes per class.
    """

    class_points, class_mask, class_id = extract_class_points(points, labels, class_name)

    if class_points.shape[0] == 0:
        print(f"No points found for class: {class_name}")
        return None, None, [], class_mask, None

    remaining_points = class_points.astype(np.float32).copy()

    generalized_points_list = []
    generalized_labels_list = []
    random_points_list = []
    plane_models = []

    print(f"Generalizing {class_name} with up to {max_planes_per_class} planes...")
    print(f"  Original class points: {class_points.shape[0]}")

    for plane_idx in range(max_planes_per_class):
        if remaining_points.shape[0] < max(min_plane_points, 3):
            print(
                f"  Stop: only {remaining_points.shape[0]} remaining points, "
                f"minimum is {min_plane_points}."
            )
            break

        plane_model, inliers = fit_plane_ransac(
            remaining_points,
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000,
        )

        if plane_model is None or len(inliers) < min_plane_points:
            print(
                f"  Stop: plane {plane_idx + 1} has {len(inliers) if inliers is not None else 0} "
                f"inliers, minimum is {min_plane_points}."
            )
            break

        inliers = np.asarray(inliers, dtype=np.int32)
        inlier_points = remaining_points[inliers]

        projected_original_points = project_points_to_plane(inlier_points, plane_model)
        random_points = np.empty((0, 3), dtype=np.float32)

        if generate_random_points:
            # Distribute the requested random points across the detected planes.
            points_for_this_plane = max(1, int(random_points_per_class // max_planes_per_class))

            random_points = generate_random_points_on_plane_region(
                projected_original_points,
                plane_model,
                num_points=points_for_this_plane,
                random_seed=random_seed + class_id * 1000 + plane_idx,
            )

        if random_points.shape[0] > 0:
            projected_points = np.vstack([projected_original_points, random_points]).astype(np.float32)
            random_points_list.append(random_points.astype(np.float32))
        else:
            projected_points = projected_original_points.astype(np.float32)

        projected_labels = np.full(
            projected_points.shape[0],
            class_id,
            dtype=np.int32,
        )

        generalized_points, generalized_labels = downsample_points_with_labels(
            projected_points,
            projected_labels,
            voxel_size=voxel_size,
        )

        generalized_points_list.append(generalized_points)
        generalized_labels_list.append(generalized_labels)
        plane_models.append(plane_model)

        print(f"  Plane {plane_idx + 1:02d}:")
        print(f"    Inlier points:       {len(inliers)}")
        print(f"    Random generated:    {random_points.shape[0]}")
        print(f"    Output points:       {generalized_points.shape[0]}")
        print(
            "    Plane model:         "
            f"a={plane_model[0]:.6f}, "
            f"b={plane_model[1]:.6f}, "
            f"c={plane_model[2]:.6f}, "
            f"d={plane_model[3]:.6f}"
        )

        keep_remaining_mask = np.ones(remaining_points.shape[0], dtype=bool)
        keep_remaining_mask[inliers] = False
        remaining_points = remaining_points[keep_remaining_mask]

    if not generalized_points_list:
        print(f"  No accepted planes for class: {class_name}")
        return None, None, [], class_mask, None

    all_generalized_points = np.vstack(generalized_points_list).astype(np.float32)
    all_generalized_labels = np.concatenate(generalized_labels_list).astype(np.int32)

    if random_points_list:
        all_random_points = np.vstack(random_points_list).astype(np.float32)
    else:
        all_random_points = np.empty((0, 3), dtype=np.float32)

    print(f"  Accepted planes:       {len(plane_models)}")
    print(f"  Total output points:   {all_generalized_points.shape[0]}")
    print(f"  Total random points:   {all_random_points.shape[0]}")

    return all_generalized_points, all_generalized_labels, plane_models, class_mask, all_random_points


def generalize_floor_wall_ceiling(
    points,
    labels,
    distance_threshold=0.08,
    voxel_size=0.10,
    keep_generalized_only=False,
    generate_random_points=False,
    random_points_per_class=5000,
    random_seed=7,
    max_planes_per_class=10,
    min_plane_points=300,
):
    generalized_points_list = []
    generalized_labels_list = []
    random_points_list = []

    generalized_class_ids = set()

    for class_name in GENERALIZE_CLASSES:
        class_id = S3DIS_CLASS_NAMES.index(class_name)
        generalized_class_ids.add(class_id)

        generalized_points, generalized_labels, plane_models, class_mask, random_points = generalize_planar_class(
            points,
            labels,
            class_name,
            distance_threshold=distance_threshold,
            voxel_size=voxel_size,
            generate_random_points=generate_random_points,
            random_points_per_class=random_points_per_class,
            random_seed=random_seed,
            max_planes_per_class=max_planes_per_class,
            min_plane_points=min_plane_points,
        )

        if generalized_points is not None:
            generalized_points_list.append(generalized_points)
            generalized_labels_list.append(generalized_labels)

        if random_points is not None and random_points.shape[0] > 0:
            random_points_list.append(random_points.astype(np.float32))

    if not keep_generalized_only:
        keep_mask = np.ones(labels.shape[0], dtype=bool)

        for class_id in generalized_class_ids:
            keep_mask &= labels != class_id

        remaining_points = points[keep_mask]
        remaining_labels = labels[keep_mask]

        if remaining_points.shape[0] > 0:
            generalized_points_list.append(remaining_points.astype(np.float32))
            generalized_labels_list.append(remaining_labels.astype(np.int32))

    if not generalized_points_list:
        raise RuntimeError("No generalized points were produced.")

    generalized_points = np.vstack(generalized_points_list).astype(np.float32)
    generalized_labels = np.concatenate(generalized_labels_list).astype(np.int32)

    if random_points_list:
        all_random_points = np.vstack(random_points_list).astype(np.float32)
    else:
        all_random_points = np.empty((0, 3), dtype=np.float32)

    print_class_counts(generalized_labels, title="Generalized output class counts")
    print(f"Total generated random points: {all_random_points.shape[0]}")

    return generalized_points, generalized_labels, all_random_points


def main():
    args = parse_args()

    # Import torch backend only after argument parsing.
    # This is where Open3D checks the installed PyTorch version.
    import open3d.ml.torch as ml3d

    if args.cfg_file is None:
        args.cfg_file = os.path.join(
            args.open3d_ml_root,
            "ml3d",
            "configs",
            "randlanet_s3dis.yml",
        )

    if not os.path.isfile(args.cfg_file):
        raise FileNotFoundError(f"Config file not found:\n  {args.cfg_file}")

    if not args.no_auto_download:
        auto_download_checkpoint(args.open3d_ml_root, args.ckpt)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {args.ckpt}\n"
            "Use --ckpt or allow automatic download."
        )

    verify_md5_if_known(args.ckpt)

    print("Loading config...")
    cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

    print("Creating RandLA-Net model...")
    cfg.model.num_points = args.num_points

    model = ml3d.models.RandLANet(**cfg.model)

    print("Creating semantic segmentation pipeline on CPU...")
    pipeline = ml3d.pipelines.SemanticSegmentation(
        model=model,
        dataset=None,
        device="cpu",
        batch_size=1,
        test_batch_size=1,
        val_batch_size=1,
    )

    print("Loading checkpoint...")
    pipeline.load_ckpt(ckpt_path=args.ckpt)

    print("Reading point cloud...")
    pcd = load_point_cloud(args.input)

    if args.voxel_size > 0:
        print(f"Voxel downsampling with voxel_size={args.voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    points = np.asarray(pcd.points).astype(np.float32)

    feat = make_features(pcd, points, args.feature_mode)

    if feat.shape[0] != points.shape[0] or feat.shape[1] != 3:
        raise RuntimeError(
            f"Feature shape must be Nx3. Got {feat.shape}, points {points.shape}."
        )

    data = {
        "point": points.astype(np.float32),
        "feat": feat.astype(np.float32),
        "label": np.zeros((points.shape[0],), dtype=np.int32),
    }

    print(f"Running CPU inference on {points.shape[0]} points...")
    result = pipeline.run_inference(data)

    pred_labels = result["predict_labels"].astype(np.int32)
    pred_scores = result["predict_scores"].astype(np.float32)

    print_class_counts(pred_labels, title="Predicted class counts")

    save_segmented_pcd(args.output, points, pred_labels)
    save_label_npz(args.output, points, pred_labels, pred_scores)

    generalized_points = None
    generalized_labels = None
    generated_random_points = np.empty((0, 3), dtype=np.float32)

    if args.generalize:
        print("\nRunning point generalization for floor, wall, and ceiling...")

        generalized_points, generalized_labels, generated_random_points = generalize_floor_wall_ceiling(
            points,
            pred_labels,
            distance_threshold=args.generalize_distance,
            voxel_size=args.generalize_voxel_size,
            keep_generalized_only=args.keep_generalized_only,
            generate_random_points=args.generate_random_points,
            random_points_per_class=args.random_points_per_class,
            random_seed=args.random_seed,
            max_planes_per_class=args.max_planes_per_class,
            min_plane_points=args.min_plane_points,
        )

        generalized_output = os.path.splitext(args.output)[0] + "_generalized.pcd"

        save_segmented_pcd(generalized_output, generalized_points, generalized_labels)
        save_label_npz(generalized_output, generalized_points, generalized_labels)

    if args.show_before_after:
        if not args.generate_random_points or generated_random_points.shape[0] == 0:
            print("WARNING: No generated random points are available to show. "
                "Use --generalize --generate_random_points."
            )
        else:
            show_original_and_generated(
                original_points=points,
                generated_points=generated_random_points,
                point_size=args.point_size,
                axis_size=args.axis_size,
            )


if __name__ == "__main__":
    main()

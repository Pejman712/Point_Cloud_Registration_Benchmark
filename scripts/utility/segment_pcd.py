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
    #"window",
    #"door",
    #"table",
    #"chair",
    #"sofa",
    #"bookcase",
    #"board",
    #"clutter",
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

KNOWN_MD5 = {
    "randlanet_s3dis_202201071330utc.pth": None,
    "randlanet_s3dis_202010091238.pth": "5f993ef4a52065e4f882764568e6f378",
    "randlanet_s3dis_area5_202010091333utc.pth": "44e45c8244a70e899065d1c1c4719b73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment a single PCD file using Open3D-ML RandLA-Net on CPU."
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
        help="Voxel downsampling size. Use 0 to disable.",
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
        "--visualize",
        action="store_true",
        help="Visualize the final segmented point cloud in Open3D with class labels.",
    )

    parser.add_argument(
        "--visualize_input",
        action="store_true",
        help="Visualize the input point cloud before inference.",
    )

    parser.add_argument(
        "--visualize_each_topic",
        action="store_true",
        help="Visualize each predicted S3DIS class/topic in a separate Open3D window.",
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


def save_label_npz(output_pcd_path, points, labels, scores):
    npz_path = os.path.splitext(output_pcd_path)[0] + "_labels.npz"

    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
        scores=scores.astype(np.float32),
        class_names=np.array(S3DIS_CLASS_NAMES),
    )

    print(f"Saved label data:\n  {npz_path}")


def print_class_legend(labels=None):
    """
    Print class names and colors in the terminal.
    """

    print("\nS3DIS class/topic legend:")

    if labels is None:
        used_labels = set(range(len(S3DIS_CLASS_NAMES)))
    else:
        used_labels = set(np.asarray(labels).astype(np.int32).tolist())

    for idx, name in enumerate(S3DIS_CLASS_NAMES):
        if idx not in used_labels:
            continue

        color = S3DIS_COLORS[idx]

        print(
            f"  {idx:2d}  {name:10s}  "
            f"RGB=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
        )


def create_colored_point_cloud(points, labels):
    """
    Create an Open3D point cloud colored by S3DIS predicted labels.
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    colors = S3DIS_COLORS[labels % len(S3DIS_COLORS)]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def make_legend_marker(position, color, size):
    """
    Create a colored sphere used as a legend marker.
    """

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    sphere.translate(position)
    sphere.paint_uniform_color(color.tolist())
    sphere.compute_vertex_normals()

    return sphere


def visualize_legacy_point_cloud(
    pcd,
    window_name="Open3D Point Cloud",
    point_size=2.0,
    axis_size=1.0,
):
    """
    Basic Open3D visualization without text labels.

    Used for input visualization and as fallback if O3DVisualizer is unavailable.
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


def visualize_point_cloud_with_legend(
    pcd,
    labels=None,
    window_name="Open3D Point Cloud with Class Legend",
    point_size=2.0,
    axis_size=1.0,
):
    """
    Visualize point cloud with an in-window legend.

    The legend is drawn beside the point cloud:
      colored sphere + label number + class/object name.

    This version uses the setup_camera signature supported by your Open3D build:
      setup_camera(field_of_view, center, eye, up)
    """

    if labels is None:
        used_labels = list(range(len(S3DIS_CLASS_NAMES)))
    else:
        used_labels = sorted(set(np.asarray(labels).astype(np.int32).tolist()))

    used_labels = [
        label for label in used_labels
        if 0 <= label < len(S3DIS_CLASS_NAMES)
    ]

    print_class_legend(np.asarray(used_labels, dtype=np.int32))

    if not hasattr(o3d.visualization, "O3DVisualizer"):
        print(
            "\nWARNING: o3d.visualization.O3DVisualizer is not available in this Open3D version."
        )
        print("Falling back to basic visualization. Text labels will only appear in terminal.")
        visualize_legacy_point_cloud(
            pcd,
            window_name=window_name,
            point_size=point_size,
            axis_size=axis_size,
        )
        return

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer(window_name, 1280, 720)
    vis.show_settings = True

    pcd_mat = o3d.visualization.rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = point_size

    vis.add_geometry("segmented_point_cloud", pcd, pcd_mat)

    if axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
        vis.add_geometry("axis", axis)

    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    extent = bbox.get_extent()

    max_extent = max(float(np.max(extent)), 1.0)
    diagonal = max(float(np.linalg.norm(extent)), 1.0)

    # Place legend to the right side of the cloud.
    legend_x = max_bound[0] + 0.12 * max_extent
    legend_y = min_bound[1]
    legend_z = max_bound[2]

    spacing = 0.07 * diagonal
    marker_size = 0.012 * diagonal

    if spacing <= 0:
        spacing = 0.1

    if marker_size <= 0:
        marker_size = 0.05

    for row, label in enumerate(used_labels):
        class_name = S3DIS_CLASS_NAMES[label]
        color = S3DIS_COLORS[label]

        marker_position = np.array(
            [
                legend_x,
                legend_y,
                legend_z - row * spacing,
            ],
            dtype=np.float64,
        )

        marker = make_legend_marker(
            position=marker_position,
            color=color,
            size=marker_size,
        )

        marker_mat = o3d.visualization.rendering.MaterialRecord()
        marker_mat.shader = "defaultLit"

        vis.add_geometry(
            f"legend_marker_{label}_{class_name}",
            marker,
            marker_mat,
        )

        text_position = marker_position + np.array(
            [marker_size * 3.0, 0.0, 0.0],
            dtype=np.float64,
        )

        vis.add_3d_label(
            text_position,
            f"{label}: {class_name}",
        )

    # Include both cloud and legend in the camera target.
    legend_min = np.array(
        [
            legend_x,
            legend_y,
            legend_z - max(len(used_labels) - 1, 0) * spacing,
        ],
        dtype=np.float64,
    )

    legend_max = np.array(
        [
            legend_x + 0.5 * max_extent,
            legend_y,
            legend_z,
        ],
        dtype=np.float64,
    )

    full_min = np.minimum(min_bound, legend_min)
    full_max = np.maximum(max_bound, legend_max)

    full_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=full_min,
        max_bound=full_max,
    )

    center = full_bbox.get_center().astype(np.float32)
    extent_full = full_bbox.get_extent()
    radius = max(float(np.linalg.norm(extent_full)), 1.0)

    eye = (
        center
        + np.array(
            [
                0.0,
                -1.8 * radius,
                0.8 * radius,
            ],
            dtype=np.float32,
        )
    )

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vis.setup_camera(
        60.0,
        center,
        eye,
        up,
    )

    app.add_window(vis)
    app.run()


def visualize_segmented_point_cloud(points, labels, point_size=2.0, axis_size=1.0):
    """
    Visualize the complete segmented point cloud with class legend.
    """

    segmented_pcd = create_colored_point_cloud(points, labels)

    visualize_point_cloud_with_legend(
        segmented_pcd,
        labels=labels,
        window_name="RandLA-Net S3DIS Segmentation with Class Legend",
        point_size=point_size,
        axis_size=axis_size,
    )


def visualize_each_topic_separately(points, labels, point_size=2.0, axis_size=1.0):
    """
    Visualize one predicted class/topic at a time.

    Each window shows the class name as a 3D label.
    Close the Open3D window to move to the next class.
    """

    unique_labels = np.unique(labels)

    for label in unique_labels:
        if 0 <= label < len(S3DIS_CLASS_NAMES):
            class_name = S3DIS_CLASS_NAMES[label]
        else:
            class_name = "unknown"

        mask = labels == label
        class_points = points[mask]

        if class_points.shape[0] == 0:
            continue

        class_pcd = o3d.geometry.PointCloud()
        class_pcd.points = o3d.utility.Vector3dVector(class_points.astype(np.float64))

        color = S3DIS_COLORS[label % len(S3DIS_COLORS)]
        class_colors = np.tile(color, (class_points.shape[0], 1))
        class_pcd.colors = o3d.utility.Vector3dVector(class_colors.astype(np.float64))

        print(f"\nVisualizing topic/class: {label} - {class_name}")
        print(f"Points: {class_points.shape[0]}")
        print(f"Color: RGB=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

        visualize_point_cloud_with_legend(
            class_pcd,
            labels=np.asarray([label], dtype=np.int32),
            window_name=f"S3DIS topic/class: {label} - {class_name}",
            point_size=point_size,
            axis_size=axis_size,
        )


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

    if args.visualize_input:
        print("Visualizing input point cloud...")
        visualize_legacy_point_cloud(
            pcd,
            window_name="Input Point Cloud",
            point_size=args.point_size,
            axis_size=args.axis_size,
        )

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

    print("\nPredicted class counts:")
    unique_labels, counts = np.unique(pred_labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if 0 <= label < len(S3DIS_CLASS_NAMES):
            class_name = S3DIS_CLASS_NAMES[label]
        else:
            class_name = "unknown"

        print(f"  {label:2d}  {class_name:10s}  {count}")

    print_class_legend(pred_labels)

    colors = S3DIS_COLORS[pred_labels % len(S3DIS_COLORS)]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    ok = o3d.io.write_point_cloud(args.output, pcd, write_ascii=False)

    if not ok:
        raise RuntimeError(f"Failed to write output point cloud: {args.output}")

    print(f"\nSaved segmented point cloud:\n  {args.output}")

    save_label_npz(args.output, points, pred_labels, pred_scores)

    if args.visualize:
        visualize_segmented_point_cloud(
            points,
            pred_labels,
            point_size=args.point_size,
            axis_size=args.axis_size,
        )

    if args.visualize_each_topic:
        visualize_each_topic_separately(
            points,
            pred_labels,
            point_size=args.point_size,
            axis_size=args.axis_size,
        )


if __name__ == "__main__":
    main()
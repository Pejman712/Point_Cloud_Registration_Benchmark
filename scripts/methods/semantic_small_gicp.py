#!/usr/bin/env python3
"""
Semantic small_gicp registration.

Pipeline:
  1. Classify source and target point clouds with Open3D-ML RandLA-Net S3DIS.
  2. For each semantic class/mask present in both clouds, run small_gicp on that segment.
  3. Return a weighted average transform over all successful segment transforms.

Expected evaluate.py call:

    result = register_clouds(
        source_cloud=current_scan,
        target_cloud=previous_scan,
        init_guess=np.eye(4),
    )

Return transformation maps source into target:

    p_target = T_target_source @ p_source
"""

import hashlib
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import small_gicp


METHOD_NAME = "semantic_small_gicp"


# RandLA-Net/S3DIS label order used by Open3D-ML.
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

S3DIS_NAME_TO_LABEL = {name: idx for idx, name in enumerate(S3DIS_CLASS_NAMES)}

KNOWN_MD5 = {
    "randlanet_s3dis_202201071330utc.pth": None,
    "randlanet_s3dis_202010091238.pth": "5f993ef4a52065e4f882764568e6f378",
    "randlanet_s3dis_area5_202010091333utc.pth": "44e45c8244a70e899065d1c1c4719b73",
}


CONFIG = {
    # small_gicp parameters
    "downsampling_resolution": 0.25,
    "num_threads": 4,

    # Filtering before classification and GICP
    "remove_non_finite": True,
    "max_points": 0,          # 0 means no random limit
    "min_points": 100,

    # Semantic segmentation parameters
    "open3d_ml_root": "./lib/Open3D-ML",
    "cfg_file": None,         # None -> <open3d_ml_root>/ml3d/configs/randlanet_s3dis.yml
    "ckpt": "scripts/utility/weights/randlanet_s3dis_202201071330utc.pth",
    "auto_download_ckpt": True,
    "verify_ckpt_md5": True,
    "device": "cpu",
    "num_points": 45056,
    "classification_voxel_size": 0.0,  # 0 disables voxel downsample before inference
    "feature_mode": "auto",             # auto, zeros, geometry

    # Which semantic masks to use for segment-wise GICP.
    # Default keeps mostly structural S3DIS classes.
    # Use None or [] to use every predicted class.
    "selected_class_names": ["ceiling", "floor", "wall", "beam", "column"],
    "min_segment_points": 100,
    "min_segment_points_after_pre_filter": 100,

    # Segment transform acceptance criteria
    "max_translation_norm": 10.0,
    "min_fitness": 0.0,

    # Transform averaging
    # weight = segment source point count by default.
    # If result.fitness is finite, weight is multiplied by max(fitness, 1e-6).
    "weight_by_fitness": True,

    # Debug
    "print_debug": False,
}


_SEGMENTATION_PIPELINE = None


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_md5_if_known(path: str) -> None:
    filename = os.path.basename(path)
    expected = KNOWN_MD5.get(filename)

    if filename not in KNOWN_MD5 or expected is None:
        return

    actual = md5sum(path)
    if actual != expected:
        raise RuntimeError(
            "Checkpoint MD5 mismatch. "
            f"File={path}, expected={expected}, actual={actual}."
        )


def download_file(url: str, output_path: str, timeout: int = 60) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return False

            tmp_path = output_path + ".tmp"
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if os.path.getsize(tmp_path) < 1024 * 1024:
                os.remove(tmp_path)
                return False

            os.replace(tmp_path, output_path)
            return True

    except (urllib.error.HTTPError, urllib.error.URLError, Exception):
        return False


def find_checkpoint_url_from_model_zoo(open3d_ml_root: str, filename: str) -> Optional[str]:
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


def auto_download_checkpoint(open3d_ml_root: str, ckpt_path: str) -> None:
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 1024 * 1024:
        return

    filename = os.path.basename(ckpt_path)
    urls = []

    url_from_model_zoo = find_checkpoint_url_from_model_zoo(open3d_ml_root, filename)
    if url_from_model_zoo:
        urls.append(url_from_model_zoo)

    urls.extend(
        [
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_202201071330utc.pth",
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_202010091238.pth",
            "https://storage.googleapis.com/open3d-releases/model-zoo/"
            "randlanet_s3dis_area5_202010091333utc.pth",
        ]
    )

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]

    for url in urls:
        if download_file(url, ckpt_path):
            return

    raise FileNotFoundError(
        "Could not download Open3D-ML RandLA-Net checkpoint. "
        f"Set CONFIG['ckpt'] to a valid checkpoint. Tried filename: {filename}"
    )


def open3d_to_numpy(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(cloud.points, dtype=np.float64)

    if CONFIG["remove_non_finite"]:
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]

    max_points = int(CONFIG.get("max_points", 0))
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    return points


def point_cloud_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def make_identity_result(reason: str = "") -> Dict:
    return {
        "transformation": np.eye(4, dtype=np.float64),
        "fitness": np.nan,
        "rmse": np.nan,
        "success": False,
        "debug": {"reason": reason},
    }


def make_features(
    cloud: o3d.geometry.PointCloud,
    points: np.ndarray,
    feature_mode: str,
) -> np.ndarray:
    """
    RandLA-Net S3DIS expects 3 feature channels in addition to XYZ.
    In S3DIS these are RGB. For LiDAR-only clouds, synthesize 3 channels.
    """

    if cloud.has_colors() and feature_mode == "auto":
        feat = np.asarray(cloud.colors).astype(np.float32)
        if feat.max(initial=0.0) > 1.0:
            feat = feat / 255.0
        return feat

    if feature_mode == "zeros":
        return np.zeros((points.shape[0], 3), dtype=np.float32)

    # geometry or auto without RGB
    z = points[:, 2:3]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

    xy_dist = np.linalg.norm(points[:, :2], axis=1, keepdims=True)
    xy_dist = (xy_dist - xy_dist.min()) / (xy_dist.max() - xy_dist.min() + 1e-6)

    ones = np.ones_like(z_norm)
    return np.concatenate([z_norm, xy_dist, ones], axis=1).astype(np.float32)


def get_segmentation_pipeline():
    """
    Lazy-load Open3D-ML RandLA-Net once.
    This avoids reloading the model for every registration call.
    """

    global _SEGMENTATION_PIPELINE

    if _SEGMENTATION_PIPELINE is not None:
        return _SEGMENTATION_PIPELINE

    # Import torch backend lazily because Open3D checks PyTorch at import time.
    import open3d.ml.torch as ml3d

    open3d_ml_root = CONFIG["open3d_ml_root"]
    cfg_file = CONFIG["cfg_file"]
    ckpt = CONFIG["ckpt"]

    if cfg_file is None:
        cfg_file = os.path.join(
            open3d_ml_root,
            "ml3d",
            "configs",
            "randlanet_s3dis.yml",
        )

    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Open3D-ML config file not found: {cfg_file}")

    if CONFIG["auto_download_ckpt"]:
        auto_download_checkpoint(open3d_ml_root, ckpt)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Open3D-ML checkpoint not found: {ckpt}")

    if CONFIG["verify_ckpt_md5"]:
        verify_md5_if_known(ckpt)

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    cfg.model.num_points = int(CONFIG["num_points"])

    model = ml3d.models.RandLANet(**cfg.model)

    pipeline = ml3d.pipelines.SemanticSegmentation(
        model=model,
        dataset=None,
        device=CONFIG["device"],
        batch_size=1,
        test_batch_size=1,
        val_batch_size=1,
    )

    pipeline.load_ckpt(ckpt_path=ckpt)
    _SEGMENTATION_PIPELINE = pipeline
    return pipeline


def classify_cloud(
    cloud: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return classified points, labels, and scores.

    Note: if classification_voxel_size > 0, classification and segment GICP run on
    the voxelized cloud rather than the original full-resolution cloud. This keeps
    labels aligned with points without needing nearest-neighbor label projection.
    """

    pcd = cloud
    voxel_size = float(CONFIG.get("classification_voxel_size", 0.0))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = open3d_to_numpy(pcd).astype(np.float32)
    if len(points) < CONFIG["min_points"]:
        raise RuntimeError(f"cloud has too few points after filtering: {len(points)}")

    feat = make_features(pcd, points, CONFIG["feature_mode"])
    if feat.shape != (points.shape[0], 3):
        raise RuntimeError(f"feature shape must be Nx3, got {feat.shape}")

    data = {
        "point": points.astype(np.float32),
        "feat": feat.astype(np.float32),
        "label": np.zeros((points.shape[0],), dtype=np.int32),
    }

    pipeline = get_segmentation_pipeline()
    result = pipeline.run_inference(data)

    labels = result["predict_labels"].astype(np.int32)
    scores = result["predict_scores"].astype(np.float32)

    if labels.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"classification returned {labels.shape[0]} labels for {points.shape[0]} points"
        )

    return points.astype(np.float64), labels, scores


def selected_label_ids() -> Optional[List[int]]:
    names = CONFIG.get("selected_class_names", None)
    if not names:
        return None

    label_ids = []
    for name in names:
        if name not in S3DIS_NAME_TO_LABEL:
            raise ValueError(f"unknown S3DIS class name in CONFIG: {name}")
        label_ids.append(S3DIS_NAME_TO_LABEL[name])

    return label_ids


def run_small_gicp_on_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_guess: np.ndarray,
) -> Dict:
    """Run small_gicp directly on numpy point arrays."""

    if len(source_points) < CONFIG["min_segment_points_after_pre_filter"]:
        return make_identity_result(f"source segment too small: {len(source_points)}")

    if len(target_points) < CONFIG["min_segment_points_after_pre_filter"]:
        return make_identity_result(f"target segment too small: {len(target_points)}")

    target, target_tree = small_gicp.preprocess_points(
        target_points,
        downsampling_resolution=CONFIG["downsampling_resolution"],
        num_threads=CONFIG["num_threads"],
    )

    source, source_tree = small_gicp.preprocess_points(
        source_points,
        downsampling_resolution=CONFIG["downsampling_resolution"],
        num_threads=CONFIG["num_threads"],
    )

    result = small_gicp.align(
        target,
        source,
        target_tree,
        init_guess,
        num_threads=CONFIG["num_threads"],
    )

    T_target_source = np.asarray(result.T_target_source, dtype=np.float64)
    if T_target_source.shape != (4, 4):
        return make_identity_result(f"invalid transform shape: {T_target_source.shape}")

    if not np.isfinite(T_target_source).all():
        return make_identity_result("transform contains nan or inf")

    translation_norm = float(np.linalg.norm(T_target_source[:3, 3]))

    fitness = getattr(result, "fitness", np.nan)
    rmse = getattr(result, "error", np.nan)

    try:
        fitness = float(fitness)
    except Exception:
        fitness = np.nan

    try:
        rmse = float(rmse)
    except Exception:
        rmse = np.nan

    success = translation_norm <= CONFIG["max_translation_norm"]
    if np.isfinite(fitness):
        success = success and fitness >= CONFIG["min_fitness"]

    return {
        "transformation": T_target_source,
        "fitness": fitness,
        "rmse": rmse,
        "success": bool(success),
        "debug": {
            "source_points": int(len(source_points)),
            "target_points": int(len(target_points)),
            "translation_norm": translation_norm,
        },
    }


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return quaternion [w, x, y, z] from a 3x3 rotation matrix."""

    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""

    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_transforms(transforms: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Weighted transform average.

    Translation: weighted arithmetic mean.
    Rotation: Markley quaternion average from weighted outer products.
    """

    if len(transforms) == 0:
        return np.eye(4, dtype=np.float64)

    weights_np = np.asarray(weights, dtype=np.float64)
    weights_np = np.maximum(weights_np, 1e-12)
    weights_np = weights_np / weights_np.sum()

    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    t_avg = np.sum(translations * weights_np[:, None], axis=0)

    A = np.zeros((4, 4), dtype=np.float64)
    q_ref = None

    for T, w in zip(transforms, weights_np):
        q = rotation_matrix_to_quaternion(T[:3, :3])

        # Keep all quaternions in the same hemisphere.
        if q_ref is None:
            q_ref = q
        elif np.dot(q_ref, q) < 0.0:
            q = -q

        A += w * np.outer(q, q)

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    q_avg = eigenvectors[:, int(np.argmax(eigenvalues))]

    if q_avg[0] < 0.0:
        q_avg = -q_avg

    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = quaternion_to_rotation_matrix(q_avg)
    T_avg[:3, 3] = t_avg
    return T_avg


def register_clouds(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_guess: Optional[np.ndarray] = None,
) -> Dict:
    """
    Classify source and target, run GICP for every matching semantic segment,
    and return the weighted average transformation.
    """

    if init_guess is None:
        init_guess = np.eye(4, dtype=np.float64)
    else:
        init_guess = np.asarray(init_guess, dtype=np.float64)

    start_time = time.time()

    try:
        source_points, source_labels, source_scores = classify_cloud(source_cloud)
        target_points, target_labels, target_scores = classify_cloud(target_cloud)

        if len(source_points) < CONFIG["min_points"]:
            return make_identity_result(f"source has too few points: {len(source_points)}")

        if len(target_points) < CONFIG["min_points"]:
            return make_identity_result(f"target has too few points: {len(target_points)}")

        selected_ids = selected_label_ids()

        if selected_ids is None:
            candidate_labels = sorted(
                set(source_labels.astype(int).tolist()).intersection(
                    set(target_labels.astype(int).tolist())
                )
            )
        else:
            candidate_labels = [
                label_id
                for label_id in selected_ids
                if np.any(source_labels == label_id) and np.any(target_labels == label_id)
            ]

        segment_results = []
        accepted_transforms = []
        accepted_weights = []

        min_segment_points = int(CONFIG["min_segment_points"])

        for label_id in candidate_labels:
            source_mask = source_labels == label_id
            target_mask = target_labels == label_id

            src = source_points[source_mask]
            tgt = target_points[target_mask]

            class_name = (
                S3DIS_CLASS_NAMES[label_id]
                if 0 <= label_id < len(S3DIS_CLASS_NAMES)
                else "unknown"
            )

            if len(src) < min_segment_points or len(tgt) < min_segment_points:
                segment_results.append(
                    {
                        "label": int(label_id),
                        "class_name": class_name,
                        "source_points": int(len(src)),
                        "target_points": int(len(tgt)),
                        "success": False,
                        "reason": "too few segment points",
                    }
                )
                continue

            result = run_small_gicp_on_points(src, tgt, init_guess)

            seg_debug = {
                "label": int(label_id),
                "class_name": class_name,
                "source_points": int(len(src)),
                "target_points": int(len(tgt)),
                "success": bool(result["success"]),
                "fitness": result.get("fitness", np.nan),
                "rmse": result.get("rmse", np.nan),
                "debug": result.get("debug", {}),
            }
            segment_results.append(seg_debug)

            if not result["success"]:
                continue

            T = np.asarray(result["transformation"], dtype=np.float64)

            weight = float(len(src))
            fitness = result.get("fitness", np.nan)
            if CONFIG["weight_by_fitness"] and np.isfinite(fitness):
                weight *= max(float(fitness), 1e-6)

            accepted_transforms.append(T)
            accepted_weights.append(weight)

        if len(accepted_transforms) == 0:
            return make_identity_result("no successful semantic segment registrations")

        T_target_source = average_transforms(accepted_transforms, accepted_weights)

        if not np.isfinite(T_target_source).all():
            return make_identity_result("average transform contains nan or inf")

        translation_norm = float(np.linalg.norm(T_target_source[:3, 3]))
        success = translation_norm <= CONFIG["max_translation_norm"]

        accepted_fitness = [
            r["fitness"] for r in segment_results
            if r.get("success") and np.isfinite(r.get("fitness", np.nan))
        ]
        accepted_rmse = [
            r["rmse"] for r in segment_results
            if r.get("success") and np.isfinite(r.get("rmse", np.nan))
        ]

        fitness = float(np.mean(accepted_fitness)) if accepted_fitness else np.nan
        rmse = float(np.mean(accepted_rmse)) if accepted_rmse else np.nan

        if np.isfinite(fitness):
            success = success and fitness >= CONFIG["min_fitness"]

        runtime_sec = time.time() - start_time

        if CONFIG["print_debug"]:
            print(
                f"[{METHOD_NAME}] "
                f"source_points={len(source_points)} "
                f"target_points={len(target_points)} "
                f"candidate_segments={len(candidate_labels)} "
                f"accepted_segments={len(accepted_transforms)} "
                f"translation_norm={translation_norm:.6f} "
                f"fitness={fitness} "
                f"rmse={rmse} "
                f"runtime={runtime_sec:.3f}s"
            )
            for r in segment_results:
                print(
                    "  segment "
                    f"{r['label']:2d} {r['class_name']:10s} "
                    f"src={r['source_points']} tgt={r['target_points']} "
                    f"success={r['success']} fitness={r.get('fitness')} rmse={r.get('rmse')}"
                )

        return {
            "transformation": T_target_source,
            "fitness": fitness,
            "rmse": rmse,
            "success": bool(success),
            "debug": {
                "source_points": int(len(source_points)),
                "target_points": int(len(target_points)),
                "source_class_counts": {
                    S3DIS_CLASS_NAMES[int(k)] if 0 <= int(k) < len(S3DIS_CLASS_NAMES) else str(int(k)): int(v)
                    for k, v in zip(*np.unique(source_labels, return_counts=True))
                },
                "target_class_counts": {
                    S3DIS_CLASS_NAMES[int(k)] if 0 <= int(k) < len(S3DIS_CLASS_NAMES) else str(int(k)): int(v)
                    for k, v in zip(*np.unique(target_labels, return_counts=True))
                },
                "candidate_segments": int(len(candidate_labels)),
                "accepted_segments": int(len(accepted_transforms)),
                "segment_results": segment_results,
                "segment_weights": [float(w) for w in accepted_weights],
                "translation_norm": translation_norm,
                "runtime_sec": runtime_sec,
                "downsampling_resolution": CONFIG["downsampling_resolution"],
                "classification_voxel_size": CONFIG["classification_voxel_size"],
                "num_threads": CONFIG["num_threads"],
            },
        }

    except Exception as e:
        return make_identity_result(f"semantic small_gicp failed: {e}")

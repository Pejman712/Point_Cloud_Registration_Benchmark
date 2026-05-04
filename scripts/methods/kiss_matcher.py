#!/usr/bin/env python3
import time
import numpy as np


METHOD_NAME = "kiss_matcher"


CONFIG = {
    # Main KISS-Matcher parameter
    "resolution": 0.5,

    # Success threshold from KISS-Matcher example
    "min_final_inliers": 5,

    # Optional filtering
    "remove_non_finite": True,
    "max_points": 0,          # 0 = no random subsampling
    "min_points": 100,

    # Optional yaw augmentation, normally keep disabled for benchmarking
    "yaw_aug_angle_deg": None,

    # Safety check
    "max_translation_norm": 50.0,

    # Debug
    "print_debug": False,
}


def cloud_to_numpy(cloud):
    points = np.asarray(cloud.points, dtype=np.float64)

    if CONFIG["remove_non_finite"]:
        # Use all(axis=1), not any(axis=1).
        # We only want rows where x, y, z are all finite.
        points = points[np.isfinite(points).all(axis=1)]

    max_points = int(CONFIG.get("max_points", 0))
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    return points


def rotate_point_cloud_yaw(points, yaw_angle_deg):
    yaw_angle_rad = np.radians(yaw_angle_deg)

    R = np.array(
        [
            [np.cos(yaw_angle_rad), -np.sin(yaw_angle_rad), 0.0],
            [np.sin(yaw_angle_rad),  np.cos(yaw_angle_rad), 0.0],
            [0.0,                    0.0,                   1.0],
        ],
        dtype=np.float64,
    )

    return (R @ points.T).T


def identity_result(reason):
    return {
        "transformation": np.eye(4, dtype=np.float64),
        "fitness": np.nan,
        "rmse": np.nan,
        "success": False,
        "debug": {
            "reason": reason,
        },
    }


def make_transform(rotation, translation):
    R = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def register_clouds(source_cloud, target_cloud, init_guess=None):
    """
    Benchmark interface.

    Input:
      source_cloud: current scan
      target_cloud: previous scan
      init_guess: 4x4 initial transform, currently unused by KISS-Matcher

    Output:
      transformation: T_target_source

    KISS-Matcher estimates a transform that maps source into target:

      p_target = R @ p_source + t

    Therefore the returned matrix is directly compatible with evaluate.py.
    """

    try:
        import kiss_matcher
    except Exception as e:
        return identity_result(f"Could not import kiss_matcher: {e}")

    start_time = time.time()

    src = cloud_to_numpy(source_cloud)
    tgt = cloud_to_numpy(target_cloud)

    if len(src) < CONFIG["min_points"]:
        return identity_result(f"source has too few points: {len(src)}")

    if len(tgt) < CONFIG["min_points"]:
        return identity_result(f"target has too few points: {len(tgt)}")

    if CONFIG["yaw_aug_angle_deg"] is not None:
        src = rotate_point_cloud_yaw(src, CONFIG["yaw_aug_angle_deg"])

    try:
        params = kiss_matcher.KISSMatcherConfig(float(CONFIG["resolution"]))
        matcher = kiss_matcher.KISSMatcher(params)

        result = matcher.estimate(src, tgt)

        R = np.asarray(result.rotation, dtype=np.float64)
        t = np.asarray(result.translation, dtype=np.float64)

        T_target_source = make_transform(R, t)

        if T_target_source.shape != (4, 4):
            return identity_result(f"invalid transform shape: {T_target_source.shape}")

        if not np.isfinite(T_target_source).all():
            return identity_result("transform contains NaN or Inf")

        num_rot_inliers = int(matcher.get_num_rotation_inliers())
        num_final_inliers = int(matcher.get_num_final_inliers())

        translation_norm = float(np.linalg.norm(T_target_source[:3, 3]))

        success = (
            num_final_inliers >= int(CONFIG["min_final_inliers"])
            and translation_norm <= float(CONFIG["max_translation_norm"])
        )

        # KISS-Matcher does not directly return Open3D-style fitness/RMSE.
        # Use final inlier ratio as a fitness-like value.
        fitness = float(num_final_inliers) / max(1.0, float(len(src)))
        rmse = np.nan

        runtime_sec = time.time() - start_time

        if CONFIG["print_debug"]:
            print(
                "[kiss_matcher] "
                f"src={len(src)} "
                f"tgt={len(tgt)} "
                f"rot_inliers={num_rot_inliers} "
                f"final_inliers={num_final_inliers} "
                f"translation_norm={translation_norm:.6f} "
                f"success={success} "
                f"runtime={runtime_sec:.3f}s"
            )

        return {
            "transformation": T_target_source,
            "fitness": fitness,
            "rmse": rmse,
            "success": bool(success),
            "debug": {
                "source_points": int(len(src)),
                "target_points": int(len(tgt)),
                "num_rotation_inliers": num_rot_inliers,
                "num_final_inliers": num_final_inliers,
                "translation_norm": translation_norm,
                "runtime_sec": runtime_sec,
                "resolution": float(CONFIG["resolution"]),
                "method": "KISS-Matcher",
            },
        }

    except Exception as e:
        return identity_result(f"KISS-Matcher failed: {e}")
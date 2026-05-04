#!/usr/bin/env python3
import time

import numpy as np
import open3d as o3d
import small_gicp


METHOD_NAME = "small_gicp"


CONFIG = {
    # Main small_gicp parameters
    "downsampling_resolution": 0.25,
    "num_threads": 4,

    # Optional cloud filtering before small_gicp
    "remove_non_finite": True,
    "max_points": 0,          # 0 means no random limit
    "min_points": 100,

    # Success criteria
    "max_translation_norm": 10.0,
    "min_fitness": 0.0,

    # Debug
    "print_debug": False,
}


def open3d_to_numpy(cloud):
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


def make_identity_result(reason=""):
    return {
        "transformation": np.eye(4, dtype=np.float64),
        "fitness": np.nan,
        "rmse": np.nan,
        "success": False,
        "debug": {
            "reason": reason,
        },
    }


def register_clouds(source_cloud, target_cloud, init_guess=None):
    """
    Register source_cloud to target_cloud.

    Expected by evaluate.py:

      result = register_clouds(
          source_cloud=current_scan,
          target_cloud=previous_scan,
          init_guess=np.eye(4),
      )

    Returns:
      {
        "transformation": T_target_source,
        "fitness": float,
        "rmse": float,
        "success": bool,
        "debug": dict
      }

    small_gicp returns result.T_target_source, which transforms source into target:

      p_target = T_target_source @ p_source

    This matches the expected output in evaluate.py.
    """

    if init_guess is None:
        init_guess = np.eye(4, dtype=np.float64)

    start_time = time.time()

    source_points = open3d_to_numpy(source_cloud)
    target_points = open3d_to_numpy(target_cloud)

    if len(source_points) < CONFIG["min_points"]:
        return make_identity_result(
            f"source has too few points: {len(source_points)}"
        )

    if len(target_points) < CONFIG["min_points"]:
        return make_identity_result(
            f"target has too few points: {len(target_points)}"
        )

    try:
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

        # API used by small_gicp examples:
        # result = small_gicp.align(target, source, target_tree, init_guess)
        result = small_gicp.align(
            target,
            source,
            target_tree,
            init_guess,
            num_threads=CONFIG["num_threads"],
        )

        T_target_source = np.asarray(result.T_target_source, dtype=np.float64)

        if T_target_source.shape != (4, 4):
            return make_identity_result(
                f"invalid transform shape: {T_target_source.shape}"
            )

        if not np.isfinite(T_target_source).all():
            return make_identity_result("transform contains nan or inf")

        translation_norm = float(np.linalg.norm(T_target_source[:3, 3]))

        success = translation_norm <= CONFIG["max_translation_norm"]

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

        if np.isfinite(fitness):
            success = success and fitness >= CONFIG["min_fitness"]

        runtime_sec = time.time() - start_time

        if CONFIG["print_debug"]:
            print(
                "[small_gicp] "
                f"source_points={len(source_points)} "
                f"target_points={len(target_points)} "
                f"translation_norm={translation_norm:.6f} "
                f"fitness={fitness} "
                f"rmse={rmse} "
                f"runtime={runtime_sec:.3f}s"
            )

        return {
            "transformation": T_target_source,
            "fitness": fitness,
            "rmse": rmse,
            "success": bool(success),
            "debug": {
                "source_points": int(len(source_points)),
                "target_points": int(len(target_points)),
                "translation_norm": translation_norm,
                "runtime_sec": runtime_sec,
                "downsampling_resolution": CONFIG["downsampling_resolution"],
                "num_threads": CONFIG["num_threads"],
            },
        }

    except Exception as e:
        return make_identity_result(f"small_gicp failed: {e}")
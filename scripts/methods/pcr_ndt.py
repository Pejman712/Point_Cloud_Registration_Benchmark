#!/usr/bin/env python3
import time
import numpy as np

from point_cloud_registration import NDT


METHOD_NAME = "pcr_ndt"


CONFIG = {
    "voxel_size": 0.5,
    "max_iter": 30,
    "max_dist": 2.0,
    "tol": 1e-3,
    "max_points": 50000,
    "min_points": 100,
    "print_debug": False,
}


def cloud_to_numpy(cloud):
    points = np.asarray(cloud.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]

    max_points = int(CONFIG["max_points"])
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    return points


def identity_result(reason):
    return {
        "transformation": np.eye(4, dtype=np.float64),
        "fitness": np.nan,
        "rmse": np.nan,
        "success": False,
        "debug": {"reason": reason},
    }


def register_clouds(source_cloud, target_cloud, init_guess=None):
    if init_guess is None:
        init_guess = np.eye(4, dtype=np.float64)

    start = time.time()

    source = cloud_to_numpy(source_cloud)
    target = cloud_to_numpy(target_cloud)

    if len(source) < CONFIG["min_points"]:
        return identity_result(f"source too small: {len(source)}")

    if len(target) < CONFIG["min_points"]:
        return identity_result(f"target too small: {len(target)}")

    try:
        reg = NDT(
            voxel_size=CONFIG["voxel_size"],
            max_iter=CONFIG["max_iter"],
            max_dist=CONFIG["max_dist"],
            tol=CONFIG["tol"],
        )

        reg.set_target(target)

        T_target_source = reg.align(source, init_T=init_guess)
        T_target_source = np.asarray(T_target_source, dtype=np.float64)

        if T_target_source.shape != (4, 4):
            return identity_result(f"invalid transform shape: {T_target_source.shape}")

        if not np.isfinite(T_target_source).all():
            return identity_result("transform has NaN or Inf")

        return {
            "transformation": T_target_source,
            "fitness": np.nan,
            "rmse": np.nan,
            "success": True,
            "debug": {
                "source_points": int(len(source)),
                "target_points": int(len(target)),
                "runtime_sec": time.time() - start,
                "method": "point_cloud_registration.NDT",
            },
        }

    except Exception as e:
        return identity_result(f"pcr_ndt failed: {e}")
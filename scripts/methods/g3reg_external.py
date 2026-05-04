#!/usr/bin/env python3
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d


METHOD_NAME = "g3reg_external"


CONFIG = {
    # Your CMake puts binaries in lib/G3Reg/bin
    "g3reg_executable": "lib/G3Reg/bin/g3reg_cli",

    # This must be relative to the G3Reg repo root because g3reg_cli
    # internally prepends config.project_path.
    "g3reg_config": "configs/hit_ms/gem_pagor.yaml",

    # Point cloud preprocessing before calling G3Reg
    "voxel_size": 0.25,
    "max_points": 80000,
    "min_points": 100,

    # External process runtime
    "timeout_sec": 120,

    # Debugging
    "keep_temp_files": False,
    "print_debug": True,
}


def repo_root():
    """
    scripts/methods/g3reg_external.py
    parents[0] = scripts/methods
    parents[1] = scripts
    parents[2] = repo root
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path):
    path = Path(str(path).strip())

    if path.is_absolute():
        return path

    return repo_root() / path


def cloud_to_clean_open3d(cloud):
    points = np.asarray(cloud.points, dtype=np.float64)

    if points.size == 0:
        return o3d.geometry.PointCloud()

    points = points[np.isfinite(points).all(axis=1)]

    max_points = int(CONFIG["max_points"])
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(points)

    voxel_size = float(CONFIG["voxel_size"])
    if voxel_size > 0:
        clean = clean.voxel_down_sample(voxel_size)

    return clean


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


def read_transform_txt(path):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"Transform file was not created: {path}")

    rows = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            vals = [float(x) for x in line.split()]
            rows.append(vals)

    T = np.asarray(rows, dtype=np.float64)

    if T.shape != (4, 4):
        raise RuntimeError(f"Invalid transform shape from {path}: {T.shape}")

    if not np.isfinite(T).all():
        raise RuntimeError("Transform contains NaN or Inf")

    return T


def write_cloud(path, cloud):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=True,
        compressed=False,
    )

    if not ok:
        raise RuntimeError(f"Failed to write PCD: {path}")


def safe_read_text(path, tail_chars=2000):
    path = Path(path)

    if not path.exists():
        return ""

    text = path.read_text(errors="replace")

    if len(text) > tail_chars:
        return text[-tail_chars:]

    return text


def register_clouds(source_cloud, target_cloud, init_guess=None):
    """
    External G3Reg wrapper.

    source_cloud:
      current scan

    target_cloud:
      previous scan

    returns:
      T_target_source, mapping source -> target

    G3Reg's example applies:
      pcl::transformPointCloud(*source, *src_transformed, tf)

    Therefore, tf maps source into target, which matches evaluate.py.
    """

    start = time.time()

    g3reg_exe = resolve_path(CONFIG["g3reg_executable"])
    g3reg_config = str(CONFIG["g3reg_config"]).strip()

    if not g3reg_exe.exists():
        return identity_result(f"G3Reg executable not found: {g3reg_exe}")

    if not g3reg_exe.is_file():
        return identity_result(f"G3Reg executable path is not a file: {g3reg_exe}")

    if not str(g3reg_exe).endswith("g3reg_cli"):
        return identity_result(f"G3Reg executable does not look like g3reg_cli: {g3reg_exe}")

    source = cloud_to_clean_open3d(source_cloud)
    target = cloud_to_clean_open3d(target_cloud)

    source_n = np.asarray(source.points).shape[0]
    target_n = np.asarray(target.points).shape[0]

    if source_n < CONFIG["min_points"]:
        return identity_result(f"source has too few points after filtering: {source_n}")

    if target_n < CONFIG["min_points"]:
        return identity_result(f"target has too few points after filtering: {target_n}")

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = Path(temp_dir_obj.name)

    try:
        source_pcd = temp_dir / "source.pcd"
        target_pcd = temp_dir / "target.pcd"
        transform_txt = temp_dir / "transform.txt"
        stdout_log = temp_dir / "g3reg_stdout.log"
        stderr_log = temp_dir / "g3reg_stderr.log"

        write_cloud(source_pcd, source)
        write_cloud(target_pcd, target)

        cmd = [
            str(g3reg_exe),
            g3reg_config,
            str(source_pcd),
            str(target_pcd),
            str(transform_txt),
        ]

        if CONFIG["print_debug"]:
            print("[g3reg_external] command:", " ".join(cmd))
            print(f"[g3reg_external] source points: {source_n}")
            print(f"[g3reg_external] target points: {target_n}")
            print(f"[g3reg_external] temp dir: {temp_dir}")

        with open(stdout_log, "w") as out_f, open(stderr_log, "w") as err_f:
            proc = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                timeout=CONFIG["timeout_sec"],
            )

        stdout_tail = safe_read_text(stdout_log)
        stderr_tail = safe_read_text(stderr_log)

        if proc.returncode != 0:
            return identity_result(
                "G3Reg failed with return code "
                f"{proc.returncode}. stdout_tail={stdout_tail}, stderr_tail={stderr_tail}"
            )

        T_target_source = read_transform_txt(transform_txt)

        runtime = time.time() - start

        return {
            "transformation": T_target_source,
            "fitness": np.nan,
            "rmse": np.nan,
            "success": True,
            "debug": {
                "source_points": int(source_n),
                "target_points": int(target_n),
                "runtime_sec": float(runtime),
                "g3reg_executable": str(g3reg_exe),
                "g3reg_config": g3reg_config,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            },
        }

    except subprocess.TimeoutExpired:
        return identity_result(f"G3Reg timed out after {CONFIG['timeout_sec']} sec")

    except Exception as e:
        return identity_result(f"G3Reg wrapper failed: {e}")

    finally:
        if CONFIG["keep_temp_files"]:
            kept_dir = repo_root() / "results" / "debug_g3reg_temp" / f"g3reg_{int(time.time())}"
            kept_dir.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copytree(temp_dir, kept_dir)
                print(f"[g3reg_external] temp files kept in: {kept_dir}")
            except Exception as e:
                print(f"[g3reg_external] failed to keep temp files: {e}")

        temp_dir_obj.cleanup()
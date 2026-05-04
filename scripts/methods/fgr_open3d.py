import numpy as np
import open3d as o3d


METHOD_NAME = "fgr_open3d"


CONFIG = {
    "voxel_size": 0.3,
    "normal_radius_multiplier": 2.0,
    "feature_radius_multiplier": 5.0,
    "max_correspondence_distance": 1.5,
    "min_fitness": 0.02,
}


def preprocess_cloud(cloud):
    voxel_size = CONFIG["voxel_size"]

    cloud_down = cloud.voxel_down_sample(voxel_size)

    cloud_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * CONFIG["normal_radius_multiplier"],
            max_nn=30,
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * CONFIG["feature_radius_multiplier"],
            max_nn=100,
        ),
    )

    return cloud_down, fpfh


def register_clouds(source_cloud, target_cloud, init_guess=None):
    source_down, source_fpfh = preprocess_cloud(source_cloud)
    target_down, target_fpfh = preprocess_cloud(target_cloud)

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=CONFIG["max_correspondence_distance"]
    )

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        option,
    )

    return {
        "transformation": result.transformation,
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "success": bool(result.fitness >= CONFIG["min_fitness"]),
    }
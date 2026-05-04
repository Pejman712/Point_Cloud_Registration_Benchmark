import numpy as np
import open3d as o3d


METHOD_NAME = "gicp_open3d"


CONFIG = {
    "voxel_size": 0.2,
    "max_correspondence_distance": 1.0,
    "max_iterations": 50,
    "min_fitness": 0.05,
}


def preprocess_cloud(cloud):
    voxel_size = CONFIG["voxel_size"]

    cloud = cloud.voxel_down_sample(voxel_size)

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3.0,
            max_nn=30,
        )
    )

    return cloud


def register_clouds(source_cloud, target_cloud, init_guess=None):
    if init_guess is None:
        init_guess = np.eye(4)

    source = preprocess_cloud(source_cloud)
    target = preprocess_cloud(target_cloud)

    result = o3d.pipelines.registration.registration_generalized_icp(
        source,
        target,
        CONFIG["max_correspondence_distance"],
        init_guess,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=CONFIG["max_iterations"]
        ),
    )

    return {
        "transformation": result.transformation,
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "success": bool(result.fitness >= CONFIG["min_fitness"]),
    }
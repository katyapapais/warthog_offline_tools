import os
import numpy as np
import numpy.linalg as npla
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "cpp/build"))

# import get_nearest_neighbors
# import extract_normals
from extrinsic_trans_alignment import extract_normal, load_aeva_pointclouds_from_dir, load_ouster_pointclouds_from_dir

def align_pointclouds(aeva_dir, ouster_dir, num_files=None):
    aeva_pc = load_aeva_pointclouds_from_dir(aeva_dir, num_files)
    ouster_pc = load_ouster_pointclouds_from_dir(ouster_dir, num_files)

    points_ae_np = np.asarray(aeva_pc.points)
    points_os_np = np.asarray(ouster_pc.points)

    # Extract normals
    points_ae_np, points_os_np = extract_normal(aeva_pc, ouster_pc)
    
    T_ouster_robot = np.loadtxt('/home/katya/ASRL/warthog_offline_tools/post_processing/calib/T_ouster_robot')
    T_aeva_robot = np.loadtxt('/home/katya/ASRL/warthog_offline_tools/post_processing/calib/T_aeva_robot')

    points_ae_np = np.hstack((points_ae_np, np.ones((points_ae_np.shape[0], 1)))) @ T_aeva_robot
    points_ae_np = points_ae_np[:, :3]
    points_os_np = np.hstack((points_os_np, np.ones((points_os_np.shape[0], 1)))) @ T_ouster_robot
    points_os_np = points_os_np[:, :3]
    aeva_pc.points = o3d.utility.Vector3dVector(points_ae_np)
    ouster_pc.points = o3d.utility.Vector3dVector(points_os_np)

    aeva_pc.paint_uniform_color([1, 0, 0])  # Red for AEVA
    ouster_pc.paint_uniform_color([0, 1, 0])  # Green for Ouster

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([aeva_pc, ouster_pc, axis], window_name="Aligned Point Clouds")

if __name__ == "__main__":
    # path = "/home/katya/ASRL/vtr3/data/rosbag2_2025_04_04-18_10_58"
    path = "/home/katya/ASRL/vtr3/data/rosbag2_2025_04_11-16_43_00"
    # path = "/home/katya/ASRL/vtr3/data/rosbag2_2025_04_11-16_39_55"
    aeva_directory = path + "/aeva"
    ouster_directory = path + "/ouster"
    num_files = None # Set to None to load all files

    align_pointclouds(aeva_directory, ouster_directory, num_files)

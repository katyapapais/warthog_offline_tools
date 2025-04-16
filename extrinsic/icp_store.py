import os
import numpy as np
import numpy.linalg as npla
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "cpp/build"))

import get_nearest_neighbors
import extract_normals

def load_aeva_pointclouds_from_dir(directory, num_files=None):
    """Load all .bin files from a directory into a single point cloud."""
    pointcloud = o3d.geometry.PointCloud()
    files = sorted(os.listdir(directory))
    if num_files is not None:
        files = files[:num_files]
    for file in files:
        if file.endswith('.bin'):
            file_path = os.path.join(directory, file)
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 10)[:, :3]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pointcloud += pc
    return pointcloud

def load_ouster_pointclouds_from_dir(directory, num_files=None):
    """Load all .bin files from a directory into a single point cloud."""
    pointcloud = o3d.geometry.PointCloud()
    files = sorted(os.listdir(directory))
    if num_files is not None:
        files = files[:num_files]
    for file in files:
        if file.endswith('.bin'):
            file_path = os.path.join(directory, file)
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 9)[:, :3]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pointcloud += pc
    return pointcloud

def icp(A, B, max_iterations=50, tolerance=1e-6, initial_guess=None):
    """
    A, B: Nx3 numpy arrays of 3D points
    initial_guess: Optional 4x4 transformation matrix to initialize the alignment
    Returns: transformation matrix (4x4), error history
    """
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[:3, :] = A.T  # aeva
    dst[:3, :] = B.T  # ouster

    prev_error = float('inf')
    error_history = []

    T = np.array([[1, 0, 0, 0.18723],
                  [0, 1, 0, 0],
                  [0, 0, 1, -0.15],
                  [0, 0, 0, 1]])
    
    src = T @ src

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations}")
        tree = KDTree(dst[:3].T)
        distances, indices = tree.query(src[:3].T)
        matched_dst = dst[:3, indices]

        # Compute transformation
        centroid_src = np.mean(src[:3], axis=1)
        centroid_dst = np.mean(matched_dst, axis=1)

        src_centered = src[:3] - centroid_src[:, None]
        dst_centered = matched_dst - centroid_dst[:, None]

        H = src_centered @ dst_centered.T
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_dst - R @ centroid_src
        T_update = np.eye(4)
        T_update[:3, :3] = R
        T_update[:3, 3] = t

        T = T_update @ T
        src = T @ np.vstack((A.T, np.ones(A.shape[0])))

        mean_error = np.mean(distances)
        error_history.append(mean_error)

        print(f"Mean error: {mean_error}")

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T

def hat(x: np.ndarray) -> np.ndarray:
  """Builds the 3x3 skew symmetric matrix

  The hat (^) operator, builds the 3x3 skew symmetric matrix from the 3x1 vector:
    v^ = [0.0  -v3   v2]
         [ v3  0.0  -v1]
         [-v2   v1  0.0]

  See eq. 5 in Barfoot-TRO-2014 for more information.

  Args:
    x (np.ndarray): a 3x1 vector
  Returns:
    np.ndarray: the 3x3 skew symmetric matrix of x
  """
  assert x.shape[-2:] == (3, 1)
  x_hat = np.zeros(x.shape[:-2] + (3, 3))
  x_hat[..., 0, 1] = -x[..., 2, 0]
  x_hat[..., 0, 2] = x[..., 1, 0]
  x_hat[..., 1, 0] = x[..., 2, 0]
  x_hat[..., 1, 2] = -x[..., 0, 0]
  x_hat[..., 2, 0] = -x[..., 1, 0]
  x_hat[..., 2, 1] = x[..., 0, 0]
  return x_hat

def _vec2rot_analytical(aaxis_ba):

  phi_ba = npla.norm(aaxis_ba, axis=-2, keepdims=True)
  axis = aaxis_ba / phi_ba

  sp = np.sin(phi_ba)
  cp = np.cos(phi_ba)

  return (cp * np.eye(3) + (1 - cp) * (axis @ axis.swapaxes(-1, -2)) + sp * hat(axis))

def _vec2rot_numerical(aaxis_ba, num_terms=10):

  C_ab = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  x_small = hat(aaxis_ba)
  x_small_n = np.tile(np.eye(3), (*aaxis_ba.shape[:-2], 1, 1))

  for n in range(1, num_terms + 1):
    x_small_n = x_small_n @ (x_small / n)
    C_ab = C_ab + x_small_n

  return C_ab

def _vec2rot(aaxis_ba, num_terms=0):
  tolerance = 1e-12
  phi_ba = npla.norm(aaxis_ba, axis=-2)[0]
  if (phi_ba < tolerance) or (num_terms != 0):
    return _vec2rot_numerical(aaxis_ba, num_terms)
  else:
    return _vec2rot_analytical(aaxis_ba)

_vec2rot_vec = np.vectorize(_vec2rot, signature='(3,1),()->(3,3)')

def vec2rot(aaxis_ba: np.ndarray, num_terms: int = 0) -> np.ndarray:
  """Builds a rotation matrix using the exponential map.

  This function builds a rotation matrix, C_ab, using the exponential map (from an axis-angle parameterization).
    C_ab = exp(aaxis_ba^),
  where aaxis_ba is a 3x1 axis-angle vector, the magnitude of the angle of rotation can be recovered by finding the norm
  of the vector, and the axis of rotation is the unit-length vector that arises from normalization.
  Note that the angle around the axis, aaxis_ba, is a right-hand-rule (counter-clockwise positive) angle from 'a' to
  'b'.
  Alternatively, we that note that
    C_ba = exp(-aaxis_ba^) = exp(aaxis_ab^).
  Typical robotics convention has some oddity when it comes using this exponential map in practice. For example, if we
  wish to integrate the kinematics:
    d/dt C = omega^ * C,
  where omega is the 3x1 angular velocity, we employ the convention:
    C_20 = exp(deltaTime*-omega^) * C_10,
  Noting that omega is negative (left-hand-rule).
  For more information see eq. 97 in Barfoot-TRO-2014.

  Args:
    aaxis_ba (np.ndarray): the axis-angle vector of the rotation
    num_terms (int): number of terms used in the infinite series approximation of the exponential map
  Returns:
    np.ndarray: the 3x3 rotation matrix of aaxis_ba
  """
  assert aaxis_ba.shape[-2:] == (3, 1)
  return _vec2rot_vec(aaxis_ba, num_terms)

def icp_new(aeva_pc, ouster_pc, max_iterations=100, tolerance=1e-14):
    theta = np.array([0, 0, 0]).reshape(3, 1)  # C_aeva_old_aeva_new
    costs = []
    thetas = []

    # Convert Open3D point cloud points to NumPy arrays
    aeva_points_np = np.asarray(aeva_pc).astype(np.float64)
    ouster_points_np = np.asarray(ouster_pc).astype(np.float64)
    matches = [None] * len(aeva_points_np)

    robust_k = 0.35
    init_steps = max_iterations - 5

    max_pair_d2 = 5.0**2

    for step in range(max_iterations):
        Cop = vec2rot(theta)
        cost = 0

        # Build (A,b) terms
        A = np.zeros((3, 3))
        b = np.zeros((3, 1))

        # Rotate aeva points (just placeholder rotation matrix for now)
        aeva_points_rot = aeva_points_np @ Cop.T

        if step < init_steps:
            correspondences = np.zeros((aeva_points_rot.shape[0], 2), dtype=np.float64)
            n_elem = np.array([aeva_points_rot.shape[0]], dtype=np.int32)

            # Call to the C++ extension using proper numpy types
            get_nearest_neighbors.get_nearest_neighbors(
                ouster_points_np,
                aeva_points_rot,
                max_pair_d2,
                correspondences,
                n_elem
            )
            matches = np.copy(correspondences)

        for i in range(matches.shape[0]):
            idx_ouster = int(matches[i, 1])
            idx_aeva = int(matches[i, 0])
            if idx_ouster >= ouster_points_np.shape[0] or idx_aeva >= aeva_points_rot.shape[0]:
                continue  # Safety check to prevent indexing errors

            xl = ouster_points_np[idx_ouster, :3].reshape(3, 1)
            xa = aeva_points_rot[idx_aeva, :3].reshape(3, 1)
            ebar = xl - xa

            u = np.linalg.norm(ebar)
            weight = 1 / (1 + (u / robust_k)**2)
            cost += np.linalg.norm(ebar)
            jac = hat(xa).T
            A += jac @ jac.T * weight
            b += -jac @ ebar * weight

        cost += np.sqrt(max_pair_d2) * (aeva_points_rot.shape[0] - matches.shape[0])

        dtheta = np.linalg.solve(A, b)
        theta = theta + dtheta

        costs.append(cost)
        thetas.append(theta[2, 0])

        print(f'step: {step:3d} cost: {cost:.2f} dtheta: {dtheta[2,0]:.6f} theta: {theta[2,0]:.4f} N: {matches.shape[0]}')

        if np.linalg.norm(dtheta) < tolerance:
            break

    return theta

def project_lidar_onto_radar(points, max_elev=0.05):
    # find points that are within the radar scan FOV
    points_out = []
    for i in range(points.shape[0]):
        elev = np.arctan2(points[i, 2], np.sqrt(points[i, 0]**2 + points[i, 1]**2))
        if np.abs(elev) <= max_elev:
            points_out.append(points[i, :])
    points = np.array(points_out)
    # project to 2D (spherical projection)
    for i in range(points.shape[0]):
        rho = np.sqrt(points[i, 0]**2 + points[i, 1]**2 + points[i, 2]**2)
        phi = np.arctan2(points[i, 1], points[i, 0])
        points[i, 0] = rho * np.cos(phi)
        points[i, 1] = rho * np.sin(phi)
        points[i, 2] = 0.0
    return points

def extract_normal(aeva_pc, ouster_pc):
    # Extract normals
    num_pts = 100000
    # OS
    points_in_os = np.asarray(ouster_pc.points)
    points_out_os = np.zeros(points_in_os.shape)
    normals_os = np.zeros(points_in_os.shape)
    normal_scores_os = np.zeros(points_in_os.shape[0])
    num_points_os = np.array([0])
    extract_normals.extract_normals(
        points_in_os,
        100,
        0.3,
        2.0,
        0.0061365,
        4.0,
        2.0,
        num_pts,
        0.95,
        points_out_os,
        normals_os,
        normal_scores_os,
        num_points_os,
    )
    points_out_os = points_out_os[:num_points_os[0], :]
    normal_scores_os = normal_scores_os[:num_points_os[0]]
    normals_os = normals_os[:num_points_os[0], :]
    # AEVA
    points_in_aeva = np.asarray(aeva_pc.points)
    points_out_aeva = np.zeros(points_in_aeva.shape)
    normals_aeva = np.zeros(points_in_aeva.shape)
    normal_scores_aeva = np.zeros(points_in_aeva.shape[0])
    num_points_aeva = np.array([0])
    extract_normals.extract_normals(
        points_in_aeva,
        100,
        0.3,
        2.0,
        0.00698132,
        4.0,
        2.0,
        num_pts,
        0.95,
        points_out_aeva,
        normals_aeva,
        normal_scores_aeva,
        num_points_aeva,
    )
    points_out_aeva = points_out_aeva[:num_points_aeva[0], :]
    normal_scores_aeva = normal_scores_aeva[:num_points_aeva[0]]
    normals_aeva = normals_aeva[:num_points_aeva[0], :]

    return points_out_aeva, points_out_os
                
def align_pointclouds(aeva_dir, ouster_dir, num_files=None):
    aeva_pc = load_aeva_pointclouds_from_dir(aeva_dir, num_files)
    ouster_pc = load_ouster_pointclouds_from_dir(ouster_dir, num_files)

    # Extract normals
    points_ex_aeva, points_ex_os = extract_normal(aeva_pc, ouster_pc)

    points_ae_np = points_ex_aeva # np.asarray(aeva_pc.points).astype(np.float64)
    points_os_np = points_ex_os # np.asarray(ouster_pc.points).astype(np.float64)

    points_out_os = project_lidar_onto_radar(points_ex_os)
    points_out_aeva = project_lidar_onto_radar(points_ex_aeva)

    # OS
    T_ouster_front = np.array([[-1, 0, 0, 0],
                               [ 0, 1, 0, 0],
                               [ 0, 0, -1, 0],
                               [ 0, 0, 0, 1]])
    T_ouster_robot = np.loadtxt('/home/katya/ASRL/warthog_offline_tools/post_processing/calib/T_ouster_robot')
    points_out_os_homogeneous = np.hstack((points_out_os, np.ones((points_out_os.shape[0], 1))))
    points_out_os_robot_frame = points_out_os_homogeneous @ T_ouster_front.T
    points_out_os = points_out_os_robot_frame[:, :3]

    # AEVA
    T_os_aeva = np.array([[1, 0, 0, 0.175],
                          [0, 1, 0, 0],
                          [0, 0, 1, -0.18],
                          [0, 0, 0, 1]])
    points_out_aeva_homogeneous = np.hstack((points_out_aeva, np.ones((points_out_aeva.shape[0], 1))))
    points_out_aeva = points_out_aeva_homogeneous @ T_os_aeva.T
    points_out_aeva = points_out_aeva[:, :3]
    
    # ICP
    theta = icp_new(points_out_aeva, points_out_os)
    T_os_aeva_update = np.eye(4)
    T_os_aeva_update[:3, :3] = vec2rot(theta)
    print("Final transformation matrix:\n", T_os_aeva_update)
    print("Final w/ Dir Calib:\n", T_ouster_front.T @ T_os_aeva_update)
    print("T_aeva_os:\n", T_ouster_front.T @ T_os_aeva_update @ T_os_aeva) 
    print("T_aeva_robot:\n", T_ouster_front.T @ T_os_aeva_update @ T_os_aeva @ T_ouster_robot)

    points_ae_np = T_os_aeva_update @ T_os_aeva @ np.hstack((points_ae_np, np.ones((points_ae_np.shape[0], 1)))).T
    points_ae_np = points_ae_np.T[:, :3]
    points_os_np = T_ouster_front @ np.hstack((points_os_np, np.ones((points_os_np.shape[0], 1)))).T
    points_os_np = points_os_np.T[:, :3]
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
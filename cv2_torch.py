import numpy as np
import cv2

import torch
from scipy.io import loadmat


def undistortPoints(uv_points, cameraMatrix, distCoeffs, iterations=5):
    """
    Iteratively undistort points to correct for lens distortion, mimicking OpenCV's cv2.undistortPoints,
    implemented in PyTorch for potential GPU acceleration.
    
    Args:
        uv_points: PyTorch tensor of distorted points, shape (N, 2).
        cameraMatrix: Camera matrix K as a PyTorch tensor.
        distCoeffs: Distortion coefficients (k1, k2, p1, p2[, k3]) as a PyTorch tensor.
        iterations: Number of iterations for the refinement process.
        device: The PyTorch device where the computations should be performed.
        
    Returns:
        Undistorted points as PyTorch tensor, shape (N, 2).
    """
    # Infer device from input tensor if not specified

    device = uv_points.device
    
    # Ensure tensors are on the correct device
    uv_points = uv_points.to(device)
    cameraMatrix = cameraMatrix.to(device)
    distCoeffs = distCoeffs.to(device)
    
    # Ensure uv_points are in the correct shape (N, 2)
    if uv_points.dim() == 3:
        uv_points = uv_points.squeeze(1)
    
    # Pre-allocate undistorted points tensor
    undistorted_points = torch.zeros_like(uv_points)
    
    # Distortion coefficients
    k1, k2, p1, p2 = distCoeffs[:4]
    k3 = distCoeffs[4] if len(distCoeffs) > 4 else 0
    
    for i in range(uv_points.shape[0]):
        # Convert to normalized camera coordinates
        x_distorted = (uv_points[i, 0] - cameraMatrix[0, 2]) / cameraMatrix[0, 0]
        y_distorted = (uv_points[i, 1] - cameraMatrix[1, 2]) / cameraMatrix[1, 1]
        
        x = x_distorted
        y = y_distorted
        
        # Iteratively solve for the undistorted coordinates
        for _ in range(iterations):
            r2 = x * x + y * y
            radial_distortion = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_delta = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_delta = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (x_distorted - x_delta) / radial_distortion
            y = (y_distorted - y_delta) / radial_distortion
        
        undistorted_points[i, 0] = x * cameraMatrix[0, 0] + cameraMatrix[0, 2]
        undistorted_points[i, 1] = y * cameraMatrix[1, 1] + cameraMatrix[1, 2]
    
    return undistorted_points

def convertPointsToHomogeneous(points):
    """
    Convert points to homogeneous coordinates.

    Args:
        points: A numpy array of shape (N, 2) for 2D points or (N, 3) for 3D points.

    Returns:
        A numpy array of points in homogeneous coordinates.
        outputs (N, 3) and (N, 4) homogeneous coordinates.
        
    """
    device = points.device
    
    # Append a column of ones to the input points
    ones = torch.ones((len(points), 1), device = device)
    homogeneous_points = torch.hstack((points, ones))
    
    return homogeneous_points

def Rodrigues(r):
    """
    Convert between a rotation vector and a rotation matrix using PyTorch.
    """
    device = r.device  # Ensures compatibility with tensors on GPU/CPU
    dtype = r.dtype  # Maintains input tensor dtype
    
    if r.numel() == 3:  # Rotation vector to matrix
        r = r.view(-1)
        theta = torch.norm(r)
        if theta < torch.finfo(dtype).eps:
            return torch.eye(3, device=device, dtype=dtype)
        
        r_normalized = r / theta
        K = torch.tensor([
            [0, -r_normalized[2], r_normalized[1]],
            [r_normalized[2], 0, -r_normalized[0]],
            [-r_normalized[1], r_normalized[0], 0]
        ], device=device, dtype=dtype)
        
        R = torch.eye(3, device=device, dtype=dtype) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)
        return R
    elif r.shape == (3, 3):  # Rotation matrix to vector
        theta = torch.arccos((torch.trace(r) - 1) / 2.0)
        if abs(theta) < torch.finfo(dtype).eps:
            return torch.zeros(3, device=device, dtype=dtype).view(3, 1)
        
        v = torch.tensor([
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1]
        ], device=device, dtype=dtype) / (2 * torch.sin(theta))
        return (theta * v).view(3, 1)
    else:
        raise ValueError("Input must be a 3-element vector or a 3x3 matrix.")
    
def projectPoints(object_points, rvec, tvec, K, dist_coeffs):
    """
    project 3D points to 2D points
    
    Args:
        object_points : N, 3 shaped 3D points with dtype float64 
        rvec : rotation vector with shape (3, 1)
        tvec : translation vector with shape (3, 1)
        K : intrinsic matrix with shape (3, 3)
        dist_coeffs : distortion coefficient shaped (4) or (5)
    """
    device = object_points.device  
    dtype = object_points.dtype 
    
    # Ensure all tensors are on the same device
    rvec = rvec.to(device=device, dtype=dtype)
    tvec = tvec.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    dist_coeffs = dist_coeffs.to(device=device, dtype=dtype)
    
    # Rodrigues transformation
    R = Rodrigues(rvec.view(-1))
    
    object_points_transf = torch.mm(object_points, R.T) + tvec.T
    
    x = object_points_transf[:, 0] / object_points_transf[:, 2]
    y = object_points_transf[:, 1] / object_points_transf[:, 2]
    
    r2 = x**2 + y**2
    
    # Check length of dist_coeffs
    if len(dist_coeffs) == 5:
        radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2 + dist_coeffs[4] * r2**3
    else:  # Assume len(dist_coeffs) == 4
        radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2

    x_distorted = x * radial_distortion + 2 * dist_coeffs[2] * x * y + dist_coeffs[3] * (r2 + 2 * x**2)
    y_distorted = y * radial_distortion + dist_coeffs[2] * (r2 + 2 * y**2) + 2 * dist_coeffs[3] * x * y
    
    x_pixel = x_distorted * K[0, 0] + K[0, 2]
    y_pixel = y_distorted * K[1, 1] + K[1, 2]
    
    return torch.stack((x_pixel, y_pixel), dim=-1)


if __name__ == "__main__":

    # Test if such functions are same as cv2 functions
    
    # Example usage for rotation vector to matrix
    # Rotation vector to matrix
    rvec = torch.tensor([0, 0, 1.57], dtype=torch.float64)  # Example rotation vector
    R = Rodrigues(rvec)  # Convert to rotation matrix
    rvec_back = Rodrigues(R)
    print("Rotation Matrix:\n", R)
    print("Rotation Vector:\n", rvec_back)

    rvec = np.array([0, 0, 1.57]).reshape(3, 1)  # Rotation vector
    R, _ = cv2.Rodrigues(rvec)  # To rotation matrix
    rvec_back, _ = cv2.Rodrigues(R)  # Back to rotation vector
    print("Rotation Matrix:\n", R)
    print("Rotation Vector:\n", rvec_back)


    # Convert inputs to PyTorch tensors and define them as needed
    # Example usage
    object_points1 = torch.tensor(np.load('./hyper_sl/utils/test_2024_04_02_20_57_30ms_depth.npy'),  dtype = torch.float64).reshape(-1, 3)
    distortion_camera = torch.tensor(loadmat('./hyper_sl/utils/distortion_camera1.mat')['distortion'], dtype = torch.float64).squeeze()
    intrinsic_camera = torch.tensor(loadmat('./hyper_sl/utils/intrinsic_camera1.mat')['K'], dtype = torch.float64)

    object_points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=torch.float64)
    rvec = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64).T  # Rotation vector as column vector
    tvec = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64).T  # Translation vector as column vector
    # camera_matrix = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    # dist_coeffs = torch.tensor([-0.2, 0.03, 0.0, 0.0, 0.0], dtype=torch.float64)

    projected_points = projectPoints(object_points1, rvec, tvec, intrinsic_camera, distortion_camera)
    # projected_points = projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    print("Projected Points:\n", projected_points)

    # Ensure inputs are in homogeneous coordinates for OpenCV as well
    object_points = np.array([[1, 1, 1], [2, 2, 1]], dtype=np.float64)
    rvec = np.array([[0, 0, 0.0]], dtype=np.float64).T
    tvec = np.array([[0, 0, 0]], dtype=np.float64).T
    # camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    # dist_coeffs = np.array([-0.2, 0.03, 0, 0, 0], dtype=np.float64)

    object_points1 = np.load('./hyper_sl/utils/test_2024_04_02_20_57_30ms_depth.npy').astype(np.float64)
    distortion_camera = np.array(loadmat('./hyper_sl/utils/distortion_camera1.mat')['distortion'], dtype = np.float64).squeeze()
    intrinsic_camera = np.array(loadmat('./hyper_sl/utils/intrinsic_camera1.mat')['K'], dtype = np.float64)

    # OpenCV projection for comparison, ensuring input points are 3D for projectPoints
    projected_points_cv2, _ = cv2.projectPoints(object_points1.reshape(-1, 1, 3), rvec, tvec, intrinsic_camera, distortion_camera)
    # projected_points_cv2, _ = cv2.projectPoints(object_points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
    print("OpenCV Projected Points:\n", projected_points_cv2.reshape(-1, 2))


    # Example usage for Homogeneous
    points_2d = torch.tensor([[2, 3], [4, 5], [6, 7]])
    points_3d = torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    homogeneous_2d = convertPointsToHomogeneous(points_2d)
    homogeneous_3d = convertPointsToHomogeneous(points_3d)
    print("Homogeneous 2D Points:\n", homogeneous_2d)
    print("Homogeneous 2D Points:\n", homogeneous_3d)

    # Example usage for 3D points
    points_2d = np.array([[2, 3], [4, 5], [6, 7]])
    points_3d = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])

    homogeneous_2d_cv2 = cv2.convertPointsToHomogeneous(points_2d)
    homogeneous_3d_cv2 = cv2.convertPointsToHomogeneous(points_3d)
    print("Homogeneous 3D Points:\n", homogeneous_2d_cv2)
    print("Homogeneous 3D Points:\n", homogeneous_3d_cv2)


    # Example For UndistortPoints
    # Convert inputs to PyTorch tensors (assuming inputs are already defined)
    uv_points = torch.tensor([[320, 240], [330, 250], [340, 260]], dtype=torch.float64)
    cameraMatrix = torch.tensor([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=torch.float64)
    distCoeffs = torch.tensor([-0.2, 0.03, 0, 0, 0], dtype=torch.float64)

    # Ours
    undistorted_points = undistortPoints(uv_points, cameraMatrix, distCoeffs)
    print("Undistorted Points:\n", undistorted_points)

    distorted_points = np.array([[320, 240], [330, 250], [340, 260]], dtype = np.float64)
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    dist_coeffs = np.array([-0.2, 0.03, 0, 0, 0])

    # OpenCV
    undistorted_points_cv2 = cv2.undistortPoints(distorted_points.reshape(-1, 1, 2), camera_matrix, dist_coeffs, None, camera_matrix).reshape(-1, 2)

    print("OpenCV Undistorted Points:\n", undistorted_points_cv2)

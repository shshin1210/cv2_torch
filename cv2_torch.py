import numpy as np
import cv2

import torch
import math
from typing import Optional, Tuple, Union

# -----------------------------
# Undistort points (iterative)
# -----------------------------
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
    
    # Distortion coefficients
    k1, k2, p1, p2 = distCoeffs[:4]
    k3 = distCoeffs[4] if len(distCoeffs) > 4 else 0
    
    # Pre-allocate undistorted points tensor
    undistorted_points = torch.zeros_like(uv_points)

    # Normalize distorted points to camera coordinates
    x_distorted = (uv_points[:, 0] - cameraMatrix[0, 2]) / cameraMatrix[0, 0]
    y_distorted = (uv_points[:, 1] - cameraMatrix[1, 2]) / cameraMatrix[1, 1]

    # Set initial guess for undistorted coordinates
    x = x_distorted.clone()
    y = y_distorted.clone()

    # Distortion coefficients
    k1, k2, p1, p2 = distCoeffs[:4]
    k3 = distCoeffs[4] if len(distCoeffs) > 4 else 0

    # Apply iterative solution to all points simultaneously
    for _ in range(iterations):
        r2 = x ** 2 + y ** 2
        radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_delta = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_delta = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
        
        x = (x_distorted - x_delta) / radial_distortion
        y = (y_distorted - y_delta) / radial_distortion

    # Reconstruct undistorted points in pixel coordinates
    undistorted_points = torch.zeros_like(uv_points)
    undistorted_points[:, 0] = x * cameraMatrix[0, 0] + cameraMatrix[0, 2]
    undistorted_points[:, 1] = y * cameraMatrix[1, 1] + cameraMatrix[1, 2]

    return undistorted_points

# -----------------------------
# Undistort points normalized (iterative)
# -----------------------------
def undistortPoints_normalized(uv_points, cameraMatrix, distCoeffs, iterations=5):
    device = uv_points.device
    uv_points = uv_points.to(device)
    cameraMatrix = cameraMatrix.to(device)
    distCoeffs = distCoeffs.to(device).reshape(-1)

    if uv_points.dim() == 3:
        uv_points = uv_points.squeeze(1)

    k1, k2, p1, p2 = distCoeffs[:4]
    k3 = distCoeffs[4] if distCoeffs.numel() > 4 else torch.tensor(0.0, device=device, dtype=uv_points.dtype)

    x_distorted = (uv_points[:, 0] - cameraMatrix[0, 2]) / cameraMatrix[0, 0]
    y_distorted = (uv_points[:, 1] - cameraMatrix[1, 2]) / cameraMatrix[1, 1]

    x = x_distorted.clone()
    y = y_distorted.clone()

    for _ in range(iterations):
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x_delta = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_delta = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        x = (x_distorted - x_delta) / radial
        y = (y_distorted - y_delta) / radial

    return torch.stack([x, y], dim=-1)

# -----------------------------
# convertPointsToHomogeneous
# -----------------------------
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

# -----------------------------
# Rodrigues and projectPoints
# -----------------------------
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
    device = object_points.device
    dtype = object_points.dtype

    rvec = rvec.to(device=device, dtype=dtype)
    tvec = tvec.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    dist_coeffs = dist_coeffs.to(device=device, dtype=dtype)

    # Handle both rotation vector and rotation matrix
    if rvec.shape == (3, 3):
        R = rvec
    elif rvec.numel() == 3:
        R = Rodrigues(rvec.reshape(3))
    else:
        raise ValueError(f"rvec must be shape (3,), (3,1), (1,3), or (3,3), but got {rvec.shape}")

    tvec = tvec.reshape(1, 3)

    object_points_transf = object_points @ R.T + tvec

    x = object_points_transf[:, 0] / object_points_transf[:, 2]
    y = object_points_transf[:, 1] / object_points_transf[:, 2]

    r2 = x**2 + y**2

    if len(dist_coeffs) == 5:
        radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2 + dist_coeffs[4] * r2**3
    else:
        radial_distortion = 1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r2**2

    x_distorted = x * radial_distortion + 2 * dist_coeffs[2] * x * y + dist_coeffs[3] * (r2 + 2 * x**2)
    y_distorted = y * radial_distortion + dist_coeffs[2] * (r2 + 2 * y**2) + 2 * dist_coeffs[3] * x * y

    x_pixel = x_distorted * K[0, 0] + K[0, 2]
    y_pixel = y_distorted * K[1, 1] + K[1, 2]

    return torch.stack((x_pixel, y_pixel), dim=-1)

# -----------------------------
# OpenCV-like enum values (remap)
# -----------------------------
INTER_NEAREST = 0
INTER_LINEAR = 1

BORDER_CONSTANT = 0
BORDER_REPLICATE = 1
BORDER_REFLECT = 2
BORDER_WRAP = 3
BORDER_REFLECT_101 = 4
BORDER_TRANSPARENT = 5

def _to_nchw(img: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    Convert image to NCHW.
    Returns:
        tensor_nchw, original_layout
    original_layout in {"HW", "HWC", "NCHW"}
    """
    if img.ndim == 2:
        return img[None, None], "HW"
    if img.ndim == 3:
        # Assume HWC if last dim is small-ish channel count
        # This matches typical image usage.
        return img.permute(2, 0, 1).unsqueeze(0), "HWC"
    if img.ndim == 4:
        return img, "NCHW"
    raise ValueError(f"Unsupported img shape: {tuple(img.shape)}")

def _from_nchw(img: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "HW":
        return img[0, 0]
    if layout == "HWC":
        return img[0].permute(1, 2, 0)
    if layout == "NCHW":
        return img
    raise ValueError(f"Unknown layout: {layout}")

def _expand_border_value(
    border_value: Union[int, float, Tuple[float, ...], list],
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(border_value, (int, float)):
        vals = [float(border_value)] * channels
    else:
        vals = list(border_value)
        if len(vals) == 1:
            vals = vals * channels
        elif len(vals) != channels:
            raise ValueError(
                f"border_value length ({len(vals)}) must be 1 or equal to channels ({channels})"
            )
    return torch.tensor(vals, dtype=dtype, device=device).view(1, channels, 1, 1)

def _border_interpolate(p: torch.Tensor, length: int, border_mode: int) -> torch.Tensor:
    """
    OpenCV-like border interpolation for integer indices.
    p: arbitrary integer tensor
    returns valid indices in [0, length-1]
    """
    if length <= 0:
        raise ValueError("length must be positive")

    if border_mode == BORDER_REPLICATE:
        return p.clamp(0, length - 1)

    if border_mode == BORDER_WRAP:
        # Python/PyTorch remainder already behaves well for negatives with this formulation
        return torch.remainder(p, length)

    if border_mode in (BORDER_REFLECT, BORDER_REFLECT_101):
        if length == 1:
            return torch.zeros_like(p)

        if border_mode == BORDER_REFLECT:
            # pattern: fedcba|abcdefgh|hgfedcb
            period = 2 * length
            p_mod = torch.remainder(p, period)
            return torch.where(p_mod < length, p_mod, period - 1 - p_mod)

        else:
            # BORDER_REFLECT_101
            # pattern: gfedcb|abcdefgh|gfedcba
            period = 2 * length - 2
            p_mod = torch.remainder(p, period)
            return torch.where(p_mod < length, p_mod, period - p_mod)

    raise ValueError(f"_border_interpolate does not support border_mode={border_mode}")

def _gather_pixels(
    src: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    border_mode: int,
    border_value: torch.Tensor,
) -> torch.Tensor:
    """
    src: (N,C,H,W)
    x,y: (N,Hout,Wout) integer index tensors
    returns sampled pixels: (N,C,Hout,Wout)
    """
    n, c, h, w = src.shape
    hout, wout = x.shape[-2], x.shape[-1]

    if border_mode == BORDER_CONSTANT or border_mode == BORDER_TRANSPARENT:
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x_safe = x.clamp(0, w - 1)
        y_safe = y.clamp(0, h - 1)
    else:
        valid = torch.ones_like(x, dtype=torch.bool)
        x_safe = _border_interpolate(x, w, border_mode)
        y_safe = _border_interpolate(y, h, border_mode)

    idx_n = torch.arange(n, device=src.device).view(n, 1, 1).expand(n, hout, wout)
    gathered = src[idx_n, :, y_safe, x_safe]  # (N,Hout,Wout,C)
    gathered = gathered.permute(0, 3, 1, 2).contiguous()  # (N,C,Hout,Wout)

    if border_mode == BORDER_CONSTANT or border_mode == BORDER_TRANSPARENT:
        valid = valid.unsqueeze(1)  # (N,1,Hout,Wout)
        gathered = torch.where(valid, gathered, border_value.expand(n, -1, hout, wout))

    return gathered

def remap(
    src: torch.Tensor,
    map1: torch.Tensor,
    map2: Optional[torch.Tensor] = None,
    interpolation: int = INTER_LINEAR,
    borderMode: int = BORDER_CONSTANT,
    borderValue: Union[int, float, Tuple[float, ...], list] = 0,
) -> torch.Tensor:
    """
    PyTorch-only implementation of OpenCV-like cv2.remap.

    Parameters
    ----------
    src : torch.Tensor
        Input image. Supported shapes:
          - (H, W)
          - (H, W, C)
          - (N, C, H, W)

    map1, map2 :
        OpenCV-like maps.
        Supported forms:
          1) map1: (Hout, Wout, 2) or (N, Hout, Wout, 2), map2=None
             map1[...,0] = x, map1[...,1] = y
          2) map1: (Hout, Wout) or (N, Hout, Wout),
             map2: (Hout, Wout) or (N, Hout, Wout)
             where map1 = x-map, map2 = y-map

    interpolation :
        INTER_NEAREST or INTER_LINEAR

    border_mode :
        BORDER_CONSTANT / REPLICATE / REFLECT / WRAP / REFLECT_101 / TRANSPARENT

    border_value :
        Scalar or per-channel value for BORDER_CONSTANT / TRANSPARENT

    Returns
    -------
    dst : torch.Tensor
        Same layout style as input src, with spatial size from map.
    """
    src_nchw, layout = _to_nchw(src)
    n, c, h, w = src_nchw.shape
    device = src_nchw.device
    dtype = src_nchw.dtype

    if map2 is None:
        if map1.ndim == 3 and map1.shape[-1] == 2:
            xmap = map1[..., 0]
            ymap = map1[..., 1]
            xmap = xmap.unsqueeze(0)  # (1,Hout,Wout)
            ymap = ymap.unsqueeze(0)
        elif map1.ndim == 4 and map1.shape[-1] == 2:
            xmap = map1[..., 0]
            ymap = map1[..., 1]
        else:
            raise ValueError(
                "When map2 is None, map1 must have shape (H,W,2) or (N,H,W,2)"
            )
    else:
        xmap = map1
        ymap = map2
        if xmap.ndim == 2:
            xmap = xmap.unsqueeze(0)
        if ymap.ndim == 2:
            ymap = ymap.unsqueeze(0)

    if xmap.ndim != 3 or ymap.ndim != 3:
        raise ValueError("xmap and ymap must have shape (N,Hout,Wout) or (Hout,Wout)")

    hout, wout = xmap.shape[-2], xmap.shape[-1]

    if xmap.shape[0] == 1 and n > 1:
        xmap = xmap.expand(n, -1, -1)
        ymap = ymap.expand(n, -1, -1)
    elif xmap.shape[0] != n:
        raise ValueError(f"Batch mismatch: src batch={n}, map batch={xmap.shape[0]}")

    xmap = xmap.to(device=device, dtype=torch.float32)
    ymap = ymap.to(device=device, dtype=torch.float32)

    border_value_t = _expand_border_value(borderValue, c, dtype, device)

    if interpolation == INTER_NEAREST:
        # OpenCV nearest is nearest-neighbor in source pixel coordinates
        x_nn = torch.round(xmap).to(torch.long)
        y_nn = torch.round(ymap).to(torch.long)

        out = _gather_pixels(src_nchw, x_nn, y_nn, borderMode, border_value_t)

        if borderMode == BORDER_TRANSPARENT:
            # OpenCV transparent means "do not modify dst" for out-of-bounds pixels.
            # Since we don't have an existing dst buffer here, a practical equivalent is:
            #   output valid samples, and keep invalid pixels as 0.
            valid = (
                (x_nn >= 0) & (x_nn < w) &
                (y_nn >= 0) & (y_nn < h)
            ).unsqueeze(1)
            zeros = torch.zeros_like(out)
            out = torch.where(valid, out, zeros)

        return _from_nchw(out, layout)

    elif interpolation == INTER_LINEAR:
        x0 = torch.floor(xmap).to(torch.long)
        y0 = torch.floor(ymap).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        wx = (xmap - x0.to(xmap.dtype)).unsqueeze(1)  # (N,1,Hout,Wout)
        wy = (ymap - y0.to(ymap.dtype)).unsqueeze(1)

        Ia = _gather_pixels(src_nchw, x0, y0, borderMode, border_value_t)  # top-left
        Ib = _gather_pixels(src_nchw, x1, y0, borderMode, border_value_t)  # top-right
        Ic = _gather_pixels(src_nchw, x0, y1, borderMode, border_value_t)  # bottom-left
        Id = _gather_pixels(src_nchw, x1, y1, borderMode, border_value_t)  # bottom-right

        wa = (1.0 - wx) * (1.0 - wy)
        wb = wx * (1.0 - wy)
        wc = (1.0 - wx) * wy
        wd = wx * wy

        out = Ia * wa + Ib * wb + Ic * wc + Id * wd

        # Match OpenCV-style dtype behavior more closely for integer images
        if src_nchw.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # OpenCV saturate_cast-like behavior
            if src_nchw.dtype == torch.uint8:
                out = out.round().clamp(0, 255).to(torch.uint8)
            elif src_nchw.dtype == torch.int8:
                out = out.round().clamp(-128, 127).to(torch.int8)
            elif src_nchw.dtype == torch.int16:
                out = out.round().clamp(-32768, 32767).to(torch.int16)
            elif src_nchw.dtype == torch.int32:
                out = out.round().clamp(-2147483648, 2147483647).to(torch.int32)
            else:
                out = out.round().to(src_nchw.dtype)
        else:
            out = out.to(src_nchw.dtype)

        if borderMode == BORDER_TRANSPARENT:
            # same note as above:
            # without an existing destination buffer, invalid output pixels are set to 0
            valid = (
                (x0 >= 0) & (x1 < w) &
                (y0 >= 0) & (y1 < h)
            ).unsqueeze(1)
            zeros = torch.zeros_like(out)
            out = torch.where(valid, out, zeros)

        return _from_nchw(out, layout)

    else:
        raise NotImplementedError(
            f"Only INTER_NEAREST ({INTER_NEAREST}) and INTER_LINEAR ({INTER_LINEAR}) are implemented."
        )

# -----------------------------
# reprojectImageTo3D
# -----------------------------
def reprojectImageTo3D(
    disparity: torch.Tensor,
    Q: torch.Tensor,
    handle_missing_values: bool = False,
    ddepth: int = -1,
) -> torch.Tensor:
    """
    PyTorch implementation of OpenCV cv2.reprojectImageTo3D.

    Parameters
    ----------
    disparity : torch.Tensor
        Supported shapes:
            - (H, W)
            - (B, H, W)
        Supported dtypes:
            uint8, int16, int32, float16, float32, float64

        Note:
            If disparity comes from OpenCV StereoBM/StereoSGBM as CV_16S,
            you usually need to divide by 16.0 before calling this function,
            just like OpenCV docs recommend.

    Q : torch.Tensor
        Shape (4, 4) or (B, 4, 4)

    handle_missing_values : bool
        If True, pixels with minimal disparity are treated as missing/outliers,
        and their Z is set to 10000.0, matching the OpenCV documentation.

    ddepth : int
        OpenCV-style output depth:
            -1 : float32 output
            3  : int16   (CV_16S)
            4  : int32   (CV_32S)
            5  : float32 (CV_32F)

    Returns
    -------
    points_3d : torch.Tensor
        Shape:
            - (H, W, 3)   if input disparity was (H, W)
            - (B, H, W, 3) if input disparity was (B, H, W)
    """
    if disparity.ndim not in (2, 3):
        raise ValueError(f"disparity must have shape (H,W) or (B,H,W), got {tuple(disparity.shape)}")

    device = disparity.device

    # Normalize input to batched form: (B, H, W)
    squeeze_batch = False
    if disparity.ndim == 2:
        disparity = disparity.unsqueeze(0)
        squeeze_batch = True

    B, H, W = disparity.shape

    # Q handling
    if Q.ndim == 2:
        if Q.shape != (4, 4):
            raise ValueError(f"Q must be (4,4) or (B,4,4), got {tuple(Q.shape)}")
        Q = Q.unsqueeze(0).expand(B, -1, -1)
    elif Q.ndim == 3:
        if Q.shape[1:] != (4, 4):
            raise ValueError(f"Q must be (4,4) or (B,4,4), got {tuple(Q.shape)}")
        if Q.shape[0] != B:
            raise ValueError(f"Batch size mismatch: disparity batch={B}, Q batch={Q.shape[0]}")
    else:
        raise ValueError(f"Q must be (4,4) or (B,4,4), got {tuple(Q.shape)}")

    # OpenCV allows integer disparity input, but the computation is geometric,
    # so do it in float32 internally.
    disp_f = disparity.to(torch.float32)
    Q = Q.to(device=device, dtype=torch.float32)

    # Pixel coordinate grid
    # OpenCV formula uses pixel coordinates directly: [x, y, d, 1]^T
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xs = xs.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    ys = ys.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)

    ones = torch.ones_like(disp_f)

    # Homogeneous input vectors: (B,H,W,4)
    homog = torch.stack([xs, ys, disp_f, ones], dim=-1)

    # Multiply by Q:
    # out[b,h,w,:] = Q[b] @ homog[b,h,w,:]
    # result shape: (B,H,W,4)
    out = torch.einsum("bij,bhwj->bhwi", Q, homog)

    X = out[..., 0]
    Y = out[..., 1]
    Z = out[..., 2]
    Wcoord = out[..., 3]

    # Divide by W
    # Match OpenCV geometric meaning. For W == 0, result can become inf/nan,
    # which is acceptable and consistent with homogeneous reprojection behavior.
    eps = torch.finfo(torch.float32).eps
    safe = torch.abs(Wcoord) > eps

    Xn = torch.where(safe, X / Wcoord, torch.full_like(X, float("inf")))
    Yn = torch.where(safe, Y / Wcoord, torch.full_like(Y, float("inf")))
    Zn = torch.where(safe, Z / Wcoord, torch.full_like(Z, float("inf")))

    points_3d = torch.stack([Xn, Yn, Zn], dim=-1)  # (B,H,W,3)

    # OpenCV docs:
    # handleMissingValues=True => pixels with minimal disparity get very large Z,
    # currently set to 10000.
    if handle_missing_values:
        min_disp = disp_f.amin(dim=(-2, -1), keepdim=True)  # (B,1,1)
        missing_mask = disp_f == min_disp
        points_3d[..., 2] = torch.where(
            missing_mask,
            torch.full_like(points_3d[..., 2], 10000.0),
            points_3d[..., 2],
        )

    # ddepth handling (OpenCV-style subset)
    # OpenCV docs say allowed outputs are CV_16S, CV_32S, CV_32F, and default is CV_32F.
    if ddepth == -1 or ddepth == 5:  # CV_32F
        points_3d = points_3d.to(torch.float32)
    elif ddepth == 3:  # CV_16S
        points_3d = torch.clamp(points_3d.round(), -32768, 32767).to(torch.int16)
    elif ddepth == 4:  # CV_32S
        points_3d = torch.clamp(
            points_3d.round(),
            -2147483648,
            2147483647,
        ).to(torch.int32)
    else:
        raise ValueError("Unsupported ddepth. Use -1, 3(CV_16S), 4(CV_32S), or 5(CV_32F).")

    if squeeze_batch:
        points_3d = points_3d[0]  # (H,W,3)

    return points_3d


# -----------------------------
# OpenCV-like enum values (initUndistortRectifyMap)
# -----------------------------
# OpenCV-like type ids commonly used for initUndistortRectifyMap
CV_32FC1 = 5
CV_32FC2 = 13

def _as_3x3(mat: Optional[torch.Tensor], name: str, device=None, dtype=torch.float32) -> torch.Tensor:
    if mat is None:
        return torch.eye(3, device=device, dtype=dtype)
    mat = torch.as_tensor(mat, device=device, dtype=dtype)
    if mat.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {tuple(mat.shape)}")
    return mat

def _parse_dist_coeffs(
    dist_coeffs: Optional[torch.Tensor],
    device,
    dtype=torch.float32,
) -> Tuple[torch.Tensor, ...]:
    """
    Parse OpenCV-style distortion coefficients.
    Supported lengths: 4, 5, 8, 12, 14

    Order:
      [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]
    Missing values are filled with zero.
    """
    if dist_coeffs is None:
        dc = torch.zeros(14, device=device, dtype=dtype)
    else:
        dc = torch.as_tensor(dist_coeffs, device=device, dtype=dtype).flatten()
        if dc.numel() not in (4, 5, 8, 12, 14):
            raise ValueError(
                f"distCoeffs length must be one of 4,5,8,12,14, got {dc.numel()}"
            )
        if dc.numel() < 14:
            dc = torch.cat([dc, torch.zeros(14 - dc.numel(), device=device, dtype=dtype)], dim=0)

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = dc
    return k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y

def _compute_tilt_projection_matrix(
    tau_x: torch.Tensor,
    tau_y: torch.Tensor,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Implements the tilt-projection part used by OpenCV distortion model.

    This follows the model represented in the OpenCV docs, where a tilt rotation
    is followed by projection using a matrix built from elements of R(tau_x, tau_y).
    """
    cx = torch.cos(tau_x)
    sx = torch.sin(tau_x)
    cy = torch.cos(tau_y)
    sy = torch.sin(tau_y)

    Rx = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, cx.item(), sx.item()],
         [0.0, -sx.item(), cx.item()]],
        device=device,
        dtype=dtype,
    )

    Ry = torch.tensor(
        [[cy.item(), 0.0, -sy.item()],
         [0.0, 1.0, 0.0],
         [sy.item(), 0.0, cy.item()]],
        device=device,
        dtype=dtype,
    )

    R_tilt = Ry @ Rx

    # OpenCV docs show:
    # [R33, 0, -R13;
    #  0, R33, -R23;
    #  0, 0, 1] * R_tilt * [x'', y'', 1]^T
    R13 = R_tilt[0, 2]
    R23 = R_tilt[1, 2]
    R33 = R_tilt[2, 2]

    P_tilt = torch.tensor(
        [[R33.item(), 0.0, -R13.item()],
         [0.0, R33.item(), -R23.item()],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    return P_tilt @ R_tilt

def initUndistortRectifyMap(
    cameraMatrix: torch.Tensor,
    distCoeffs: Optional[torch.Tensor],
    R: Optional[torch.Tensor],
    newCameraMatrix: Optional[torch.Tensor],
    size: Tuple[int, int],
    m1type: int = CV_32FC1,
):
    """
    PyTorch implementation of cv2.initUndistortRectifyMap.

    Parameters
    ----------
    camera_matrix : (3,3)
        Original camera intrinsics.
    dist_coeffs : (N,), optional
        OpenCV distortion coefficients with N in {4,5,8,12,14}.
    R : (3,3), optional
        Rectification transform. If None, identity is used.
    new_camera_matrix : (3,3), optional
        New intrinsics for rectified image. If None, camera_matrix is used.
    size : (width, height)
        Output map size, same convention as OpenCV.
    m1type : int
        Supported:
          - CV_32FC1 -> returns (map1, map2), each (H,W) float32
          - CV_32FC2 -> returns (map1, None), map1 is (H,W,2) float32

    Returns
    -------
    map1, map2
        OpenCV-like remap inputs.
    """
    width, height = size
    device = cameraMatrix.device if isinstance(cameraMatrix, torch.Tensor) else "cpu"
    dtype = torch.float32

    K = _as_3x3(cameraMatrix, "cameraMatrix", device=device, dtype=dtype)
    R = _as_3x3(R, "R", device=device, dtype=dtype)
    P = _as_3x3(newCameraMatrix, "newCameraMatrix", device=device, dtype=dtype) if newCameraMatrix is not None else K

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    fx_new = P[0, 0]
    fy_new = P[1, 1]
    cx_new = P[0, 2]
    cy_new = P[1, 2]

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = _parse_dist_coeffs(
        distCoeffs, device=device, dtype=dtype
    )

    # Destination pixel grid
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )

    # Step 1: pixel in rectified image -> normalized coords under new camera matrix
    x = (xs - cx_new) / fx_new
    y = (ys - cy_new) / fy_new
    ones = torch.ones_like(x)

    # Step 2: apply inverse rectification
    R_inv = torch.linalg.inv(R)
    xyz = torch.stack([x, y, ones], dim=-1)                 # (H,W,3)
    xyz_unrect = torch.einsum("ij,hwj->hwi", R_inv, xyz)    # (H,W,3)

    X = xyz_unrect[..., 0]
    Y = xyz_unrect[..., 1]
    W = xyz_unrect[..., 2]

    x1 = X / W
    y1 = Y / W

    # Step 3: distortion model
    r2 = x1 * x1 + y1 * y1
    r4 = r2 * r2
    r6 = r4 * r2

    radial_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_num / radial_den

    x2 = (
        x1 * radial
        + 2 * p1 * x1 * y1
        + p2 * (r2 + 2 * x1 * x1)
        + s1 * r2
        + s2 * r4
    )
    y2 = (
        y1 * radial
        + p1 * (r2 + 2 * y1 * y1)
        + 2 * p2 * x1 * y1
        + s3 * r2
        + s4 * r4
    )

    # Step 4: tilt distortion if present
    if torch.abs(tau_x) > 0 or torch.abs(tau_y) > 0:
        tilt_mat = _compute_tilt_projection_matrix(tau_x, tau_y, device=device, dtype=dtype)
        xy1 = torch.stack([x2, y2, torch.ones_like(x2)], dim=-1)  # (H,W,3)
        tilted = torch.einsum("ij,hwj->hwi", tilt_mat, xy1)
        x3 = tilted[..., 0] / tilted[..., 2]
        y3 = tilted[..., 1] / tilted[..., 2]
    else:
        x3 = x2
        y3 = y2

    # Step 5: project using original camera matrix
    map_x = fx * x3 + cx
    map_y = fy * y3 + cy

    if m1type == CV_32FC1:
        return map_x.to(torch.float32), map_y.to(torch.float32)

    if m1type == CV_32FC2:
        map1 = torch.stack([map_x, map_y], dim=-1).to(torch.float32)
        return map1, None

    raise NotImplementedError(
        "Supported m1type values are CV_32FC1 and CV_32FC2 only. "
        "CV_16SC2 fixed-point maps are not implemented in this pure PyTorch version."
    )


# -----------------------------
# OpenCV-like enum values (stereoRectify)
# -----------------------------
CALIB_ZERO_DISPARITY = 0x400

def _as_torch(x, dtype=torch.float64, device=None):
    return torch.as_tensor(x, dtype=dtype, device=device)

def _rodrigues_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rvec)
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    if theta < 1e-15:
        return I
    k = rvec / theta
    kx, ky, kz = k
    K = torch.tensor([
        [0.0, -kz,  ky],
        [kz,  0.0, -kx],
        [-ky, kx,  0.0]
    ], dtype=rvec.dtype, device=rvec.device)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

def _matrix_to_rodrigues(R: torch.Tensor) -> torch.Tensor:
    tr = torch.trace(R)
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-15:
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    denom = 2.0 * torch.sin(theta)
    rx = (R[2, 1] - R[1, 2]) / denom
    ry = (R[0, 2] - R[2, 0]) / denom
    rz = (R[1, 0] - R[0, 1]) / denom
    axis = torch.stack([rx, ry, rz])
    return axis * theta

def _prepare_dist(dist, dtype, device):
    if dist is None:
        return torch.zeros(14, dtype=dtype, device=device)
    d = _as_torch(dist, dtype=dtype, device=device).reshape(-1)
    if d.numel() < 14:
        d = torch.cat([d, torch.zeros(14 - d.numel(), dtype=dtype, device=device)])
    return d[:14]

def _undistort_points_iter(
    pts_px: torch.Tensor,   # (N,2)
    K: torch.Tensor,
    dist: torch.Tensor,
    iters: int = 8,
) -> torch.Tensor:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y = dist

    x = (pts_px[:, 0] - cx) / fx
    y = (pts_px[:, 1] - cy) / fy
    x0 = x.clone()
    y0 = y.clone()

    for _ in range(iters):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
        radial_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_den

        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r4
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r4

        x = (x0 - dx) / radial
        y = (y0 - dy) / radial

    return torch.stack([x, y], dim=1)

def _rectify_points_normalized(
    pts_px: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    R_rect: torch.Tensor,
) -> torch.Tensor:
    und = _undistort_points_iter(pts_px, K, dist)  # normalized undistorted
    xyz = torch.cat(
        [und, torch.ones((und.shape[0], 1), dtype=und.dtype, device=und.device)],
        dim=1,
    )  # (N,3)

    xyz_r = (R_rect @ xyz.T).T
    x = xyz_r[:, 0] / xyz_r[:, 2]
    y = xyz_r[:, 1] / xyz_r[:, 2]
    return torch.stack([x, y], dim=1)

def _corner_grid(image_size, dtype, device):
    w, h = image_size
    return torch.tensor([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [0.0, h - 1.0],
        [w - 1.0, h - 1.0],
    ], dtype=dtype, device=device)

def stereoRectify(
    cameraMatrix1,
    distCoeffs1,
    cameraMatrix2,
    distCoeffs2,
    imageSize: Tuple[int, int],
    R,
    T,
    flags: int = CALIB_ZERO_DISPARITY,
    alpha: float = -1.0,
    newImageSize: Optional[Tuple[int, int]] = None,
):
    """
    Closer-to-OpenCV stereoRectify in pure PyTorch.

    Returns:
        R1, R2, P1, P2, Q, roi1, roi2
    """
    dtype = torch.float64
    device = _as_torch(cameraMatrix1).device if isinstance(cameraMatrix1, torch.Tensor) else torch.device("cpu")

    K1 = _as_torch(cameraMatrix1, dtype=dtype, device=device)
    K2 = _as_torch(cameraMatrix2, dtype=dtype, device=device)
    R = _as_torch(R, dtype=dtype, device=device).reshape(3, 3)
    T = _as_torch(T, dtype=dtype, device=device).reshape(3)

    D1 = _prepare_dist(distCoeffs1, dtype, device)
    D2 = _prepare_dist(distCoeffs2, dtype, device)

    w, h = imageSize
    if newImageSize is None or newImageSize == (0, 0):
        new_w, new_h = w, h
    else:
        new_w, new_h = newImageSize

    # ---- 1) OpenCV-style common orientation ----
    rvec = _matrix_to_rodrigues(R)
    r_r = _rodrigues_to_matrix(-0.5 * rvec)
    t = r_r @ T

    idx = 0 if abs(t[0]) > abs(t[1]) else 1
    c = 1.0 if t[idx] > 0 else -1.0

    uu = torch.zeros(3, dtype=dtype, device=device)
    uu[idx] = c

    ww = torch.cross(t, uu, dim=0)
    nw = torch.linalg.norm(ww)
    nt = torch.linalg.norm(t)

    if nw > 1e-15:
        ww = ww * (torch.acos(torch.abs(t[idx]) / nt) / nw)
        wR = _rodrigues_to_matrix(ww)
    else:
        wR = torch.eye(3, dtype=dtype, device=device)

    # IMPORTANT: these two were wrong before
    R1 = wR @ r_r.T
    R2 = wR @ r_r

    # Rectified translation
    t_rect = R2 @ T

    # ---- 2) choose focal length ----
    sx = new_w / w
    sy = new_h / h

    fx1 = K1[0, 0] * sx
    fy1 = K1[1, 1] * sy
    fx2 = K2[0, 0] * sx
    fy2 = K2[1, 1] * sy

    # OpenCV uses a common focal length; for horizontal stereo this is driven
    # by y direction, for vertical stereo by x direction.
    fc_new = min(fy1, fy2) if idx == 0 else min(fx1, fx2)

    # ---- 3) compute new principal points from rectified corners ----
    corners = _corner_grid((w, h), dtype, device)

    pts1n = _rectify_points_normalized(corners, K1, D1, R1)
    pts2n = _rectify_points_normalized(corners, K2, D2, R2)

    mean1 = pts1n.mean(dim=0)
    mean2 = pts2n.mean(dim=0)

    cc1 = torch.empty(2, dtype=dtype, device=device)
    cc2 = torch.empty(2, dtype=dtype, device=device)

    cc1[0] = (new_w - 1) * 0.5 - fc_new * mean1[0]
    cc1[1] = (new_h - 1) * 0.5 - fc_new * mean1[1]
    cc2[0] = (new_w - 1) * 0.5 - fc_new * mean2[0]
    cc2[1] = (new_h - 1) * 0.5 - fc_new * mean2[1]

    if flags & CALIB_ZERO_DISPARITY:
        # match principal points on the non-disparity axis
        if idx == 0:  # horizontal stereo
            cy = 0.5 * (cc1[1] + cc2[1])
            cc1[1] = cy
            cc2[1] = cy
            cx = 0.5 * (cc1[0] + cc2[0])
            cc1[0] = cx
            cc2[0] = cx
        else:         # vertical stereo
            cx = 0.5 * (cc1[0] + cc2[0])
            cc1[0] = cx
            cc2[0] = cx
            cy = 0.5 * (cc1[1] + cc2[1])
            cc1[1] = cy
            cc2[1] = cy

    # ---- 4) build P1/P2 ----
    P1 = torch.zeros((3, 4), dtype=dtype, device=device)
    P2 = torch.zeros((3, 4), dtype=dtype, device=device)

    P1[0, 0] = fc_new
    P1[1, 1] = fc_new
    P1[0, 2] = cc1[0]
    P1[1, 2] = cc1[1]
    P1[2, 2] = 1.0

    P2[0, 0] = fc_new
    P2[1, 1] = fc_new
    P2[0, 2] = cc2[0]
    P2[1, 2] = cc2[1]
    P2[2, 2] = 1.0
    P2[idx, 3] = t_rect[idx] * fc_new

    # ---- 5) alpha scaling (simple but tied to final P) ----
    # Keep this conservative; alpha mainly affects focal scaling / cropping.
    if alpha >= 0:
        def projected_rectified_bounds(K, D, Rr, cc):
            ptsn = _rectify_points_normalized(corners, K, D, Rr)
            u = fc_new * ptsn[:, 0] + cc[0]
            v = fc_new * ptsn[:, 1] + cc[1]
            return u.min(), u.max(), v.min(), v.max()

        u1min, u1max, v1min, v1max = projected_rectified_bounds(K1, D1, R1, cc1)
        u2min, u2max, v2min, v2max = projected_rectified_bounds(K2, D2, R2, cc2)

        umin = min(u1min, u2min)
        umax = max(u1max, u2max)
        vmin = min(v1min, v2min)
        vmax = max(v1max, v2max)

        s0 = max(
            (new_w - 1) / max(umax - umin, 1e-12),
            (new_h - 1) / max(vmax - vmin, 1e-12),
        )
        s1 = min(
            (new_w - 1) / max(umax - umin, 1e-12),
            (new_h - 1) / max(vmax - vmin, 1e-12),
        )
        s = s0 * (1.0 - alpha) + s1 * alpha

        P1[0, 0] *= s
        P1[1, 1] *= s
        P2[0, 0] *= s
        P2[1, 1] *= s
        P2[idx, 3] *= s

        fc_new = P1[0, 0]

    # ---- 6) Q from final P1/P2 ----
    Q = torch.zeros((4, 4), dtype=dtype, device=device)

    if idx == 0:
        Tx = P2[0, 3] / P2[0, 0]
        Q[0, 0] = 1.0
        Q[0, 3] = -P1[0, 2]
        Q[1, 1] = 1.0
        Q[1, 3] = -P1[1, 2]
        Q[2, 3] = P1[0, 0]
        Q[3, 2] = -1.0 / Tx
        Q[3, 3] = (P1[0, 2] - P2[0, 2]) / Tx
    else:
        Ty = P2[1, 3] / P2[1, 1]
        Q[0, 0] = 1.0
        Q[0, 3] = -P1[0, 2]
        Q[1, 1] = 1.0
        Q[1, 3] = -P1[1, 2]
        Q[2, 3] = P1[1, 1]
        Q[3, 2] = -1.0 / Ty
        Q[3, 3] = (P1[1, 2] - P2[1, 2]) / Ty

    # ---- 7) lightweight ROI ----
    def compute_roi(K, D, Rr, P):
        ptsn = _rectify_points_normalized(corners, K, D, Rr)
        u = P[0, 0] * ptsn[:, 0] + P[0, 2]
        v = P[1, 1] * ptsn[:, 1] + P[1, 2]
        xmin = max(0, int(torch.ceil(u.min()).item()))
        xmax = min(new_w - 1, int(torch.floor(u.max()).item()))
        ymin = max(0, int(torch.ceil(v.min()).item()))
        ymax = min(new_h - 1, int(torch.floor(v.max()).item()))
        return (xmin, ymin, max(0, xmax - xmin + 1), max(0, ymax - ymin + 1))

    roi1 = compute_roi(K1, D1, R1, P1)
    roi2 = compute_roi(K2, D2, R2, P2)

    return (
        R1.to(torch.float32),
        R2.to(torch.float32),
        P1.to(torch.float32),
        P2.to(torch.float32),
        Q.to(torch.float32),
        roi1,
        roi2,
    )


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
    object_points = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]], dtype=torch.float64)
    rvec = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64).T  # Rotation vector as column vector
    tvec = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64).T  # Translation vector as column vector
    camera_matrix = torch.tensor([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    dist_coeffs = torch.tensor([-0.2, 0.03, 0.0, 0.0, 0.0], dtype=torch.float64)

    projected_points = projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    print("Projected Points:\n", projected_points)

    # Ensure inputs are in homogeneous coordinates for OpenCV as well
    object_points = np.array([[1, 1, 1], [2, 2, 1]], dtype=np.float64)
    rvec = np.array([[0, 0, 0.0]], dtype=np.float64).T
    tvec = np.array([[0, 0, 0]], dtype=np.float64).T
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([-0.2, 0.03, 0, 0, 0], dtype=np.float64)

    # OpenCV projection for comparison, ensuring input points are 3D for projectPoints
    projected_points_cv2, _ = cv2.projectPoints(object_points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs)
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
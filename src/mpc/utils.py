"""Utility functions for MPC applications."""

import random
import numpy as np
import torch
from typing import Optional, Union


def setup_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_fallback() -> str:
    """Get the best available device for computations.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to [-pi, pi].
    
    Args:
        angle: Angle(s) to normalize
        
    Returns:
        Normalized angle(s)
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle to [0, 2*pi].
    
    Args:
        angle: Angle(s) to wrap
        
    Returns:
        Wrapped angle(s)
    """
    return np.mod(angle, 2 * np.pi)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector.
    
    Args:
        v: 3D vector
        
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis-angle representation.
    
    Args:
        axis: Rotation axis (unit vector)
        angle: Rotation angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    axis = axis / np.linalg.norm(axis)
    K = skew_symmetric(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        
    Returns:
        3x3 rotation matrix
    """
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return R_z @ R_y @ R_x


def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
        
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        
    return np.array([w, x, y, z])


def compute_trajectory_error(
    reference: np.ndarray,
    actual: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """Compute trajectory tracking error.
    
    Args:
        reference: Reference trajectory (T x n)
        actual: Actual trajectory (T x n)
        weights: Optional weights for different dimensions
        
    Returns:
        Root mean square error
    """
    error = reference - actual
    
    if weights is not None:
        error = error * weights
        
    rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))
    return rmse


def compute_control_effort(control_trajectory: np.ndarray) -> float:
    """Compute control effort (sum of squared controls).
    
    Args:
        control_trajectory: Control trajectory (T x m)
        
    Returns:
        Total control effort
    """
    return np.sum(np.sum(control_trajectory**2, axis=1))


def compute_smoothness(trajectory: np.ndarray) -> float:
    """Compute trajectory smoothness (sum of squared accelerations).
    
    Args:
        trajectory: Trajectory (T x n)
        
    Returns:
        Smoothness metric
    """
    if trajectory.shape[0] < 3:
        return 0.0
        
    # Compute second differences (approximate acceleration)
    acc = np.diff(trajectory, n=2, axis=0)
    smoothness = np.sum(np.sum(acc**2, axis=1))
    
    return smoothness


def interpolate_trajectory(
    trajectory: np.ndarray,
    new_length: int
) -> np.ndarray:
    """Interpolate trajectory to new length.
    
    Args:
        trajectory: Original trajectory (T x n)
        new_length: Desired length
        
    Returns:
        Interpolated trajectory
    """
    T, n = trajectory.shape
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, new_length)
    
    interpolated = np.zeros((new_length, n))
    for i in range(n):
        interpolated[:, i] = np.interp(t_new, t_old, trajectory[:, i])
        
    return interpolated


def generate_reference_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    duration: float,
    dt: float,
    trajectory_type: str = "linear"
) -> np.ndarray:
    """Generate reference trajectory.
    
    Args:
        start: Start state
        goal: Goal state
        duration: Trajectory duration
        dt: Time step
        trajectory_type: Type of trajectory ("linear", "smooth")
        
    Returns:
        Reference trajectory
    """
    n_steps = int(duration / dt)
    n_dims = len(start)
    
    if trajectory_type == "linear":
        # Linear interpolation
        trajectory = np.zeros((n_steps, n_dims))
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            trajectory[i] = (1 - alpha) * start + alpha * goal
            
    elif trajectory_type == "smooth":
        # Smooth trajectory using cubic spline
        t = np.linspace(0, duration, n_steps)
        trajectory = np.zeros((n_steps, n_dims))
        
        for i in range(n_dims):
            # Simple smooth interpolation
            s = t / duration
            trajectory[:, i] = start[i] + (goal[i] - start[i]) * (3*s**2 - 2*s**3)
            
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
    return trajectory

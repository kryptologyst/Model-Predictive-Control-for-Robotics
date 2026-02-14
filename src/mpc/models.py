"""Robot models and system dynamics for MPC applications."""

from typing import Optional, Tuple, Callable, Union
import numpy as np
import casadi as ca
from abc import ABC, abstractmethod


class SystemModel(ABC):
    """Abstract base class for system models."""
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute system dynamics.
        
        Args:
            state: Current state vector
            control: Control input vector
            
        Returns:
            Next state vector
        """
        pass
        
    @abstractmethod
    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize system around operating point.
        
        Args:
            state: Operating point state
            control: Operating point control
            
        Returns:
            Tuple of (A, B) matrices for linearized system
        """
        pass


class LinearSystem(SystemModel):
    """Linear system model: x(k+1) = A*x(k) + B*u(k)."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray) -> None:
        """Initialize linear system.
        
        Args:
            A: State transition matrix (nx x nx)
            B: Control input matrix (nx x nu)
        """
        self.A = A
        self.B = B
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute linear dynamics."""
        return self.A @ state + self.B @ control
        
    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearization is trivial for linear systems."""
        return self.A, self.B


class NonlinearSystem(SystemModel):
    """Nonlinear system model with symbolic dynamics."""
    
    def __init__(
        self,
        dynamics_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        state_dim: int,
        control_dim: int
    ) -> None:
        """Initialize nonlinear system.
        
        Args:
            dynamics_function: Function f(x, u) -> x_next
            state_dim: State dimension
            control_dim: Control dimension
        """
        self.dynamics_function = dynamics_function
        self.nx = state_dim
        self.nu = control_dim
        
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute nonlinear dynamics."""
        return self.dynamics_function(state, control)
        
    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize system using numerical differentiation."""
        eps = 1e-6
        
        # Compute A matrix (df/dx)
        A = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            state_plus = state.copy()
            state_plus[i] += eps
            state_minus = state.copy()
            state_minus[i] -= eps
            
            f_plus = self.dynamics_function(state_plus, control)
            f_minus = self.dynamics_function(state_minus, control)
            A[:, i] = (f_plus - f_minus) / (2 * eps)
            
        # Compute B matrix (df/du)
        B = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            control_plus = control.copy()
            control_plus[i] += eps
            control_minus = control.copy()
            control_minus[i] -= eps
            
            f_plus = self.dynamics_function(state, control_plus)
            f_minus = self.dynamics_function(state, control_minus)
            B[:, i] = (f_plus - f_minus) / (2 * eps)
            
        return A, B


class RobotModel(SystemModel):
    """Base class for robot models."""
    
    def __init__(self, name: str) -> None:
        """Initialize robot model.
        
        Args:
            name: Name of the robot model
        """
        self.name = name
        self.nx = 0
        self.nu = 0
        
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds for the robot."""
        return -np.inf * np.ones(self.nx), np.inf * np.ones(self.nx)
        
    def get_control_bounds(self) -> Tuple[float, float]:
        """Get control bounds for the robot."""
        return -np.inf, np.inf


class TwoLinkArm(RobotModel):
    """Two-link robotic arm model."""
    
    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        m1: float = 1.0,
        m2: float = 1.0,
        g: float = 9.81
    ) -> None:
        """Initialize two-link arm.
        
        Args:
            l1: Length of first link
            l2: Length of second link
            m1: Mass of first link
            m2: Mass of second link
            g: Gravitational acceleration
        """
        super().__init__("TwoLinkArm")
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        
        # State: [q1, q2, q1_dot, q2_dot] (joint angles and velocities)
        # Control: [tau1, tau2] (joint torques)
        self.nx = 4
        self.nu = 2
        
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute arm dynamics using Lagrangian formulation."""
        q1, q2, q1_dot, q2_dot = state
        tau1, tau2 = control
        
        # Inertia matrix
        M11 = self.m1 * self.l1**2 / 3 + self.m2 * (self.l1**2 + self.l2**2 / 3 + self.l1 * self.l2 * np.cos(q2))
        M12 = self.m2 * (self.l2**2 / 3 + self.l1 * self.l2 * np.cos(q2) / 2)
        M21 = M12
        M22 = self.m2 * self.l2**2 / 3
        
        M = np.array([[M11, M12], [M21, M22]])
        
        # Coriolis and centrifugal forces
        C1 = -self.m2 * self.l1 * self.l2 * np.sin(q2) * q2_dot * (2 * q1_dot + q2_dot) / 2
        C2 = self.m2 * self.l1 * self.l2 * np.sin(q2) * q1_dot**2 / 2
        
        C = np.array([C1, C2])
        
        # Gravity forces
        G1 = (self.m1 * self.l1 / 2 + self.m2 * self.l1) * self.g * np.cos(q1) + \
             self.m2 * self.l2 * self.g * np.cos(q1 + q2) / 2
        G2 = self.m2 * self.l2 * self.g * np.cos(q1 + q2) / 2
        
        G = np.array([G1, G2])
        
        # Compute accelerations
        tau = np.array([tau1, tau2])
        q_ddot = np.linalg.solve(M, tau - C - G)
        
        # State derivative
        state_dot = np.array([q1_dot, q2_dot, q_ddot[0], q_ddot[1]])
        
        return state + state_dot * 0.01  # Simple Euler integration with dt=0.01
        
    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize around operating point."""
        eps = 1e-6
        
        # A matrix
        A = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            state_plus = state.copy()
            state_plus[i] += eps
            state_minus = state.copy()
            state_minus[i] -= eps
            
            f_plus = self.dynamics(state_plus, control)
            f_minus = self.dynamics(state_minus, control)
            A[:, i] = (f_plus - f_minus) / (2 * eps)
            
        # B matrix
        B = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            control_plus = control.copy()
            control_plus[i] += eps
            control_minus = control.copy()
            control_minus[i] -= eps
            
            f_plus = self.dynamics(state, control_plus)
            f_minus = self.dynamics(state, control_minus)
            B[:, i] = (f_plus - f_minus) / (2 * eps)
            
        return A, B
        
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds."""
        x_min = np.array([-np.pi, -np.pi, -10.0, -10.0])
        x_max = np.array([np.pi, np.pi, 10.0, 10.0])
        return x_min, x_max
        
    def get_control_bounds(self) -> Tuple[float, float]:
        """Get control bounds."""
        return -50.0, 50.0


class DifferentialDrive(RobotModel):
    """Differential drive mobile robot model."""
    
    def __init__(self, wheel_base: float = 0.5, wheel_radius: float = 0.1) -> None:
        """Initialize differential drive robot.
        
        Args:
            wheel_base: Distance between wheels
            wheel_radius: Wheel radius
        """
        super().__init__("DifferentialDrive")
        self.L = wheel_base
        self.r = wheel_radius
        
        # State: [x, y, theta] (position and orientation)
        # Control: [v_left, v_right] (wheel velocities)
        self.nx = 3
        self.nu = 2
        
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute differential drive dynamics."""
        x, y, theta = state
        v_left, v_right = control
        
        # Convert wheel velocities to linear and angular velocities
        v = self.r * (v_left + v_right) / 2
        omega = self.r * (v_right - v_left) / self.L
        
        # State derivative
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega
        
        state_dot = np.array([x_dot, y_dot, theta_dot])
        
        return state + state_dot * 0.01  # Simple Euler integration with dt=0.01
        
    def linearize(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linearize around operating point."""
        eps = 1e-6
        
        # A matrix
        A = np.zeros((self.nx, self.nx))
        for i in range(self.nx):
            state_plus = state.copy()
            state_plus[i] += eps
            state_minus = state.copy()
            state_minus[i] -= eps
            
            f_plus = self.dynamics(state_plus, control)
            f_minus = self.dynamics(state_minus, control)
            A[:, i] = (f_plus - f_minus) / (2 * eps)
            
        # B matrix
        B = np.zeros((self.nx, self.nu))
        for i in range(self.nu):
            control_plus = control.copy()
            control_plus[i] += eps
            control_minus = control.copy()
            control_minus[i] -= eps
            
            f_plus = self.dynamics(state, control_plus)
            f_minus = self.dynamics(state, control_minus)
            B[:, i] = (f_plus - f_minus) / (2 * eps)
            
        return A, B
        
    def get_state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state bounds."""
        x_min = np.array([-10.0, -10.0, -np.pi])
        x_max = np.array([10.0, 10.0, np.pi])
        return x_min, x_max
        
    def get_control_bounds(self) -> Tuple[float, float]:
        """Get control bounds."""
        return -5.0, 5.0


def create_casadi_dynamics(model: RobotModel) -> Callable:
    """Create CasADi symbolic dynamics function for nonlinear MPC.
    
    Args:
        model: Robot model
        
    Returns:
        CasADi function for dynamics
    """
    # Symbolic variables
    x = ca.SX.sym('x', model.nx)
    u = ca.SX.sym('u', model.nu)
    
    # Create symbolic dynamics (simplified for CasADi)
    if isinstance(model, TwoLinkArm):
        q1, q2, q1_dot, q2_dot = x[0], x[1], x[2], x[3]
        tau1, tau2 = u[0], u[1]
        
        # Simplified dynamics for CasADi
        dt = 0.01
        q1_ddot = tau1 / (model.m1 * model.l1**2 / 3)
        q2_ddot = tau2 / (model.m2 * model.l2**2 / 3)
        
        x_next = ca.vertcat(
            q1 + q1_dot * dt,
            q2 + q2_dot * dt,
            q1_dot + q1_ddot * dt,
            q2_dot + q2_ddot * dt
        )
        
    elif isinstance(model, DifferentialDrive):
        x_pos, y_pos, theta = x[0], x[1], x[2]
        v_left, v_right = u[0], u[1]
        
        v = model.r * (v_left + v_right) / 2
        omega = model.r * (v_right - v_left) / model.L
        
        dt = 0.01
        x_next = ca.vertcat(
            x_pos + v * ca.cos(theta) * dt,
            y_pos + v * ca.sin(theta) * dt,
            theta + omega * dt
        )
        
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
        
    # Create CasADi function
    dynamics_func = ca.Function('dynamics', [x, u], [x_next])
    
    return lambda x, u: dynamics_func(x, u)

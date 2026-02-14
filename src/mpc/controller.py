"""Core MPC controller implementations."""

from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np
import casadi as ca
from scipy.optimize import minimize
import torch
import warnings


class MPCController:
    """Base class for Model Predictive Control implementations.
    
    This class provides the foundation for MPC controllers with proper
    type hints, error handling, and safety features.
    """
    
    def __init__(
        self,
        prediction_horizon: int,
        state_dim: int,
        control_dim: int,
        state_cost_matrix: np.ndarray,
        control_cost_matrix: np.ndarray,
        control_limits: Optional[Tuple[float, float]] = None,
        state_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: str = "cpu"
    ) -> None:
        """Initialize MPC controller.
        
        Args:
            prediction_horizon: Number of steps to predict ahead
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            state_cost_matrix: Cost matrix for state deviations (Q)
            control_cost_matrix: Cost matrix for control inputs (R)
            control_limits: Tuple of (min, max) control limits
            state_limits: Tuple of (min, max) state limits
            device: Device to use for computations ('cpu', 'cuda', 'mps')
        """
        self.N = prediction_horizon
        self.nx = state_dim
        self.nu = control_dim
        self.Q = state_cost_matrix
        self.R = control_cost_matrix
        self.control_limits = control_limits
        self.state_limits = state_limits
        self.device = device
        
        # Safety checks
        if self.Q.shape != (self.nx, self.nx):
            raise ValueError(f"Q matrix shape {self.Q.shape} doesn't match state dim {self.nx}")
        if self.R.shape != (self.nu, self.nu):
            raise ValueError(f"R matrix shape {self.R.shape} doesn't match control dim {self.nu}")
            
        # Initialize state and control trajectory
        self.current_state = np.zeros(self.nx)
        self.control_trajectory = np.zeros((self.N, self.nu))
        
    def set_state(self, state: np.ndarray) -> None:
        """Set current state of the system.
        
        Args:
            state: Current state vector
        """
        if state.shape != (self.nx,):
            raise ValueError(f"State shape {state.shape} doesn't match expected {self.nx}")
        self.current_state = state.copy()
        
    def solve(self, reference_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve MPC optimization problem.
        
        Args:
            reference_trajectory: Reference trajectory to track (N x nx)
            
        Returns:
            Tuple of (optimal_control, predicted_state)
        """
        raise NotImplementedError("Subclasses must implement solve method")
        
    def _apply_control_limits(self, control: np.ndarray) -> np.ndarray:
        """Apply control limits to control input.
        
        Args:
            control: Control input to limit
            
        Returns:
            Limited control input
        """
        if self.control_limits is not None:
            u_min, u_max = self.control_limits
            control = np.clip(control, u_min, u_max)
        return control
        
    def _apply_state_limits(self, state: np.ndarray) -> np.ndarray:
        """Apply state limits to state vector.
        
        Args:
            state: State vector to limit
            
        Returns:
            Limited state vector
        """
        if self.state_limits is not None:
            x_min, x_max = self.state_limits
            state = np.clip(state, x_min, x_max)
        return state


class LinearMPC(MPCController):
    """Linear Model Predictive Controller.
    
    Implements MPC for linear systems of the form:
    x(k+1) = A*x(k) + B*u(k)
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        prediction_horizon: int,
        state_cost_matrix: np.ndarray,
        control_cost_matrix: np.ndarray,
        control_limits: Optional[Tuple[float, float]] = None,
        state_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: str = "cpu"
    ) -> None:
        """Initialize Linear MPC controller.
        
        Args:
            A: State transition matrix (nx x nx)
            B: Control input matrix (nx x nu)
            prediction_horizon: Number of steps to predict ahead
            state_cost_matrix: Cost matrix for state deviations (Q)
            control_cost_matrix: Cost matrix for control inputs (R)
            control_limits: Tuple of (min, max) control limits
            state_limits: Tuple of (min, max) state limits
            device: Device to use for computations
        """
        super().__init__(
            prediction_horizon=prediction_horizon,
            state_dim=A.shape[0],
            control_dim=B.shape[1],
            state_cost_matrix=state_cost_matrix,
            control_cost_matrix=control_cost_matrix,
            control_limits=control_limits,
            state_limits=state_limits,
            device=device
        )
        
        self.A = A
        self.B = B
        
        # Safety checks
        if A.shape != (self.nx, self.nx):
            raise ValueError(f"A matrix shape {A.shape} doesn't match state dim {self.nx}")
        if B.shape != (self.nx, self.nu):
            raise ValueError(f"B matrix shape {B.shape} doesn't match expected ({self.nx}, {self.nu})")
            
    def solve(self, reference_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve linear MPC optimization problem.
        
        Args:
            reference_trajectory: Reference trajectory to track (N x nx)
            
        Returns:
            Tuple of (optimal_control, predicted_state)
        """
        if reference_trajectory is None:
            reference_trajectory = np.zeros((self.N, self.nx))
            
        # Set up bounds for optimization
        bounds = []
        for _ in range(self.N):
            if self.control_limits is not None:
                bounds.append(self.control_limits)
            else:
                bounds.append((None, None))
                
        # Initial guess (previous solution or zeros)
        u_init = self.control_trajectory.flatten()
        
        # Solve optimization
        result = minimize(
            fun=self._objective_function,
            x0=u_init,
            args=(reference_trajectory,),
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if not result.success:
            warnings.warn(f"MPC optimization failed: {result.message}")
            # Use previous solution or zeros as fallback
            u_opt = self.control_trajectory[0]
        else:
            u_opt = result.x.reshape(self.N, self.nu)[0]
            
        # Predict next state
        next_state = self.A @ self.current_state + self.B @ u_opt
        next_state = self._apply_state_limits(next_state)
        
        # Update control trajectory
        self.control_trajectory = result.x.reshape(self.N, self.nu)
        
        return u_opt, next_state
        
    def _objective_function(self, u_flat: np.ndarray, reference: np.ndarray) -> float:
        """Objective function for MPC optimization.
        
        Args:
            u_flat: Flattened control trajectory
            reference: Reference trajectory
            
        Returns:
            Cost function value
        """
        u = u_flat.reshape(self.N, self.nu)
        x = self.current_state.copy()
        cost = 0.0
        
        for k in range(self.N):
            # State cost
            state_error = x - reference[k]
            cost += state_error.T @ self.Q @ state_error
            
            # Control cost
            cost += u[k].T @ self.R @ u[k]
            
            # Predict next state
            if k < self.N - 1:
                x = self.A @ x + self.B @ u[k]
                x = self._apply_state_limits(x)
                
        return cost


class NonlinearMPC(MPCController):
    """Nonlinear Model Predictive Controller using CasADi.
    
    Implements MPC for nonlinear systems using symbolic optimization.
    """
    
    def __init__(
        self,
        dynamics_function,
        prediction_horizon: int,
        state_dim: int,
        control_dim: int,
        state_cost_matrix: np.ndarray,
        control_cost_matrix: np.ndarray,
        control_limits: Optional[Tuple[float, float]] = None,
        state_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        device: str = "cpu"
    ) -> None:
        """Initialize Nonlinear MPC controller.
        
        Args:
            dynamics_function: Function that computes x(k+1) = f(x(k), u(k))
            prediction_horizon: Number of steps to predict ahead
            state_dim: Dimension of state vector
            control_dim: Dimension of control vector
            state_cost_matrix: Cost matrix for state deviations (Q)
            control_cost_matrix: Cost matrix for control inputs (R)
            control_limits: Tuple of (min, max) control limits
            state_limits: Tuple of (min, max) state limits
            device: Device to use for computations
        """
        super().__init__(
            prediction_horizon=prediction_horizon,
            state_dim=state_dim,
            control_dim=control_dim,
            state_cost_matrix=state_cost_matrix,
            control_cost_matrix=control_cost_matrix,
            control_limits=control_limits,
            state_limits=state_limits,
            device=device
        )
        
        self.dynamics_function = dynamics_function
        self._setup_optimization_problem()
        
    def _setup_optimization_problem(self) -> None:
        """Set up the CasADi optimization problem."""
        # Decision variables
        self.X = ca.SX.sym('X', self.nx, self.N + 1)  # States
        self.U = ca.SX.sym('U', self.nu, self.N)      # Controls
        
        # Parameters
        self.x0 = ca.SX.sym('x0', self.nx)  # Initial state
        self.ref = ca.SX.sym('ref', self.nx, self.N)  # Reference trajectory
        
        # Cost function
        cost = 0
        g = []  # Constraints
        
        # Initial condition constraint
        g.append(self.X[:, 0] - self.x0)
        
        # Dynamics constraints and cost
        for k in range(self.N):
            # State cost
            state_error = self.X[:, k] - self.ref[:, k]
            cost += state_error.T @ self.Q @ state_error
            
            # Control cost
            cost += self.U[:, k].T @ self.R @ self.U[:, k]
            
            # Dynamics constraint
            x_next = self.dynamics_function(self.X[:, k], self.U[:, k])
            g.append(self.X[:, k + 1] - x_next)
            
        # Terminal cost
        state_error = self.X[:, -1] - self.ref[:, -1]
        cost += state_error.T @ self.Q @ state_error
        
        # Create NLP
        nlp = {
            'x': ca.vertcat(self.X.reshape((-1, 1)), self.U.reshape((-1, 1))),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(self.x0, self.ref.reshape((-1, 1)))
        }
        
        # Create solver
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 1000,
                'tol': 1e-6
            }
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
    def solve(self, reference_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve nonlinear MPC optimization problem.
        
        Args:
            reference_trajectory: Reference trajectory to track (N x nx)
            
        Returns:
            Tuple of (optimal_control, predicted_state)
        """
        if reference_trajectory is None:
            reference_trajectory = np.zeros((self.N, self.nx))
            
        # Set up bounds
        lbx = []
        ubx = []
        
        # State bounds
        if self.state_limits is not None:
            x_min, x_max = self.state_limits
            for _ in range(self.N + 1):
                lbx.extend(x_min)
                ubx.extend(x_max)
        else:
            lbx.extend([-ca.inf] * self.nx * (self.N + 1))
            ubx.extend([ca.inf] * self.nx * (self.N + 1))
            
        # Control bounds
        if self.control_limits is not None:
            u_min, u_max = self.control_limits
            for _ in range(self.N):
                lbx.extend([u_min] * self.nu)
                ubx.extend([u_max] * self.nu)
        else:
            lbx.extend([-ca.inf] * self.nu * self.N)
            ubx.extend([ca.inf] * self.nu * self.N)
            
        # Initial guess
        x_init = np.tile(self.current_state, self.N + 1)
        u_init = np.zeros(self.nu * self.N)
        x0 = np.concatenate([x_init, u_init])
        
        # Parameters
        p = np.concatenate([self.current_state, reference_trajectory.flatten()])
        
        # Solve
        sol = self.solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            p=p
        )
        
        if self.solver.stats()['success']:
            # Extract solution
            X_opt = sol['x'][:self.nx * (self.N + 1)].reshape(self.nx, self.N + 1)
            U_opt = sol['x'][self.nx * (self.N + 1):].reshape(self.nu, self.N)
            
            u_opt = U_opt[:, 0]
            next_state = X_opt[:, 1]
            
            # Update control trajectory
            self.control_trajectory = U_opt.T
            
        else:
            warnings.warn("Nonlinear MPC optimization failed")
            u_opt = np.zeros(self.nu)
            next_state = self.current_state
            
        return u_opt, next_state

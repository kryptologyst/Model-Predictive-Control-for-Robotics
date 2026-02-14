"""Unit tests for MPC framework."""

import pytest
import numpy as np
from typing import Dict, Any

from mpc.controller import LinearMPC, NonlinearMPC
from mpc.models import TwoLinkArm, DifferentialDrive, create_casadi_dynamics
from mpc.evaluation import MPCEvaluator, Leaderboard, EvaluationMetrics
from mpc.config import MPCConfig, ModelConfig, ExperimentConfig
from mpc.utils import setup_seed, device_fallback, normalize_angle


class TestLinearMPC:
    """Test cases for LinearMPC controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        setup_seed(42)
        
        # Simple double integrator system
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0.5], [1]])
        
        self.mpc = LinearMPC(
            A=self.A,
            B=self.B,
            prediction_horizon=5,
            state_cost_matrix=np.diag([1, 1]),
            control_cost_matrix=np.array([[0.1]]),
            control_limits=(-2.0, 2.0)
        )
        
    def test_initialization(self):
        """Test MPC initialization."""
        assert self.mpc.N == 5
        assert self.mpc.nx == 2
        assert self.mpc.nu == 1
        assert self.mpc.control_limits == (-2.0, 2.0)
        
    def test_set_state(self):
        """Test state setting."""
        state = np.array([1.0, 2.0])
        self.mpc.set_state(state)
        np.testing.assert_array_equal(self.mpc.current_state, state)
        
    def test_set_state_invalid_shape(self):
        """Test state setting with invalid shape."""
        with pytest.raises(ValueError):
            self.mpc.set_state(np.array([1.0]))  # Wrong dimension
            
    def test_solve_basic(self):
        """Test basic MPC solve."""
        self.mpc.set_state(np.array([0.0, 0.0]))
        reference = np.array([1.0, 0.0])
        
        u_opt, x_next = self.mpc.solve(reference_trajectory=np.array([reference]))
        
        assert isinstance(u_opt, np.ndarray)
        assert isinstance(x_next, np.ndarray)
        assert u_opt.shape == (1,)
        assert x_next.shape == (2,)
        
    def test_control_limits(self):
        """Test control limit enforcement."""
        self.mpc.set_state(np.array([0.0, 0.0]))
        reference = np.array([10.0, 0.0])  # Large reference to trigger limits
        
        u_opt, _ = self.mpc.solve(reference_trajectory=np.array([reference]))
        
        # Control should be within limits
        assert -2.0 <= u_opt[0] <= 2.0
        
    def test_objective_function(self):
        """Test objective function computation."""
        u_test = np.array([0.5, 0.3, 0.1, -0.1, -0.3])
        self.mpc.set_state(np.array([0.0, 0.0]))
        
        cost = self.mpc._objective_function(u_test, np.array([[1.0, 0.0]]))
        
        assert isinstance(cost, float)
        assert cost >= 0  # Cost should be non-negative


class TestNonlinearMPC:
    """Test cases for NonlinearMPC controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        setup_seed(42)
        
        self.arm_model = TwoLinkArm()
        self.dynamics_func = create_casadi_dynamics(self.arm_model)
        
        self.mpc = NonlinearMPC(
            dynamics_function=self.dynamics_func,
            prediction_horizon=5,
            state_dim=self.arm_model.nx,
            control_dim=self.arm_model.nu,
            state_cost_matrix=np.diag([1, 1, 0.1, 0.1]),
            control_cost_matrix=np.diag([0.01, 0.01]),
            control_limits=(-10.0, 10.0)
        )
        
    def test_initialization(self):
        """Test nonlinear MPC initialization."""
        assert self.mpc.N == 5
        assert self.mpc.nx == 4
        assert self.mpc.nu == 2
        
    def test_solve_basic(self):
        """Test basic nonlinear MPC solve."""
        self.mpc.set_state(np.array([0.0, 0.0, 0.0, 0.0]))
        reference = np.array([[np.pi/4, np.pi/6, 0.0, 0.0]] * 5)
        
        u_opt, x_next = self.mpc.solve(reference_trajectory=reference)
        
        assert isinstance(u_opt, np.ndarray)
        assert isinstance(x_next, np.ndarray)
        assert u_opt.shape == (2,)
        assert x_next.shape == (4,)


class TestRobotModels:
    """Test cases for robot models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        setup_seed(42)
        
    def test_two_link_arm(self):
        """Test two-link arm model."""
        arm = TwoLinkArm(l1=1.0, l2=1.0, m1=1.0, m2=1.0)
        
        assert arm.nx == 4
        assert arm.nu == 2
        
        # Test dynamics
        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0])
        
        next_state = arm.dynamics(state, control)
        assert next_state.shape == (4,)
        
        # Test linearization
        A, B = arm.linearize(state, control)
        assert A.shape == (4, 4)
        assert B.shape == (4, 2)
        
    def test_differential_drive(self):
        """Test differential drive model."""
        robot = DifferentialDrive(wheel_base=0.5, wheel_radius=0.1)
        
        assert robot.nx == 3
        assert robot.nu == 2
        
        # Test dynamics
        state = np.array([0.0, 0.0, 0.0])
        control = np.array([1.0, 1.0])
        
        next_state = robot.dynamics(state, control)
        assert next_state.shape == (3,)
        
        # Test linearization
        A, B = robot.linearize(state, control)
        assert A.shape == (3, 3)
        assert B.shape == (3, 2)


class TestEvaluation:
    """Test cases for evaluation framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        setup_seed(42)
        
    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        metrics = EvaluationMetrics(
            rmse_position=0.1,
            rmse_velocity=0.05,
            control_effort=10.0,
            success_rate=1.0
        )
        
        assert metrics.rmse_position == 0.1
        assert metrics.success_rate == 1.0
        
    def test_leaderboard(self):
        """Test leaderboard functionality."""
        leaderboard = Leaderboard()
        
        metrics = EvaluationMetrics(rmse_position=0.1, success_rate=1.0)
        leaderboard.add_result("TestController", "TestCase", metrics)
        
        df = leaderboard.get_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["controller"] == "TestController"
        assert df.iloc[0]["rmse_position"] == 0.1


class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_mpc_config(self):
        """Test MPC configuration."""
        config = MPCConfig(
            prediction_horizon=10,
            state_cost_matrix=[[1.0, 0.0], [0.0, 1.0]],
            control_cost_matrix=[[0.1]]
        )
        
        assert config.prediction_horizon == 10
        assert config.state_cost_matrix == [[1.0, 0.0], [0.0, 1.0]]
        
        # Test conversion to dict
        config_dict = config.to_dict()
        assert config_dict["prediction_horizon"] == 10
        
    def test_experiment_config(self):
        """Test experiment configuration."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            seed=123
        )
        
        assert config.experiment_name == "test_experiment"
        assert config.seed == 123


class TestUtils:
    """Test cases for utility functions."""
    
    def test_setup_seed(self):
        """Test seed setup."""
        setup_seed(42)
        
        # Test that seeds are set (basic check)
        np.random.seed(42)
        val1 = np.random.random()
        
        setup_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
        
    def test_device_fallback(self):
        """Test device fallback."""
        device = device_fallback()
        assert device in ["cpu", "cuda", "mps"]
        
    def test_normalize_angle(self):
        """Test angle normalization."""
        # Test basic normalization
        angle = 3 * np.pi
        normalized = normalize_angle(angle)
        assert -np.pi <= normalized <= np.pi
        
        # Test array normalization
        angles = np.array([3 * np.pi, -3 * np.pi, np.pi/2])
        normalized = normalize_angle(angles)
        assert np.all(normalized >= -np.pi)
        assert np.all(normalized <= np.pi)


if __name__ == "__main__":
    pytest.main([__file__])

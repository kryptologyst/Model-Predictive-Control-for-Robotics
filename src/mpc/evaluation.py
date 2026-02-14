"""Evaluation framework for MPC controllers."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import warnings


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    rmse_position: float = 0.0
    rmse_velocity: float = 0.0
    rmse_control: float = 0.0
    control_effort: float = 0.0
    smoothness: float = 0.0
    settling_time: float = 0.0
    overshoot: float = 0.0
    computation_time: float = 0.0
    success_rate: float = 0.0
    constraint_violations: int = 0


class Evaluator(ABC):
    """Abstract base class for MPC evaluators."""
    
    @abstractmethod
    def evaluate(
        self,
        controller,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, EvaluationMetrics]:
        """Evaluate controller on test cases.
        
        Args:
            controller: MPC controller to evaluate
            test_cases: List of test case dictionaries
            
        Returns:
            Dictionary mapping test case names to metrics
        """
        pass


class MPCEvaluator(Evaluator):
    """Evaluator for MPC controllers."""
    
    def __init__(
        self,
        simulation_time: float = 10.0,
        dt: float = 0.01,
        position_tolerance: float = 0.1,
        velocity_tolerance: float = 0.1
    ) -> None:
        """Initialize MPC evaluator.
        
        Args:
            simulation_time: Total simulation time
            dt: Time step
            position_tolerance: Position error tolerance for success
            velocity_tolerance: Velocity error tolerance for success
        """
        self.simulation_time = simulation_time
        self.dt = dt
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        
    def evaluate(
        self,
        controller,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, EvaluationMetrics]:
        """Evaluate MPC controller on test cases.
        
        Args:
            controller: MPC controller to evaluate
            test_cases: List of test case dictionaries
            
        Returns:
            Dictionary mapping test case names to metrics
        """
        results = {}
        
        for test_case in test_cases:
            name = test_case["name"]
            print(f"Evaluating test case: {name}")
            
            metrics = self._evaluate_single_case(controller, test_case)
            results[name] = metrics
            
        return results
        
    def _evaluate_single_case(
        self,
        controller,
        test_case: Dict[str, Any]
    ) -> EvaluationMetrics:
        """Evaluate controller on a single test case."""
        # Extract test case parameters
        initial_state = test_case["initial_state"]
        reference_trajectory = test_case["reference_trajectory"]
        model = test_case["model"]
        
        # Initialize controller
        controller.set_state(initial_state)
        
        # Storage for simulation data
        n_steps = int(self.simulation_time / self.dt)
        state_history = np.zeros((n_steps, model.nx))
        control_history = np.zeros((n_steps, model.nu))
        reference_history = np.zeros((n_steps, model.nx))
        
        # Initialize
        current_state = initial_state.copy()
        state_history[0] = current_state
        reference_history[0] = reference_trajectory[0]
        
        # Simulation loop
        computation_times = []
        constraint_violations = 0
        
        for k in range(1, n_steps):
            # Get reference for current time step
            ref_idx = min(k, len(reference_trajectory) - 1)
            current_reference = reference_trajectory[ref_idx]
            reference_history[k] = current_reference
            
            # Solve MPC
            start_time = time.time()
            try:
                control_input, predicted_state = controller.solve(
                    reference_trajectory=reference_trajectory[ref_idx:ref_idx+controller.N]
                )
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                # Check for constraint violations
                if controller.control_limits is not None:
                    u_min, u_max = controller.control_limits
                    if np.any(control_input < u_min) or np.any(control_input > u_max):
                        constraint_violations += 1
                        
            except Exception as e:
                warnings.warn(f"MPC solve failed at step {k}: {e}")
                control_input = np.zeros(model.nu)
                computation_time = 0.0
                computation_times.append(computation_time)
                
            # Apply control and simulate
            current_state = model.dynamics(current_state, control_input)
            
            # Store data
            state_history[k] = current_state
            control_history[k] = control_input
            
        # Compute metrics
        metrics = self._compute_metrics(
            state_history,
            control_history,
            reference_history,
            computation_times,
            constraint_violations
        )
        
        return metrics
        
    def _compute_metrics(
        self,
        state_history: np.ndarray,
        control_history: np.ndarray,
        reference_history: np.ndarray,
        computation_times: List[float],
        constraint_violations: int
    ) -> EvaluationMetrics:
        """Compute evaluation metrics."""
        # Position and velocity errors (assuming first half is position, second half is velocity)
        n_states = state_history.shape[1]
        n_positions = n_states // 2
        
        position_error = state_history[:, :n_positions] - reference_history[:, :n_positions]
        velocity_error = state_history[:, n_positions:] - reference_history[:, n_positions:]
        
        rmse_position = np.sqrt(np.mean(np.sum(position_error**2, axis=1)))
        rmse_velocity = np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))
        
        # Control error (assuming zero reference control)
        rmse_control = np.sqrt(np.mean(np.sum(control_history**2, axis=1)))
        
        # Control effort
        control_effort = np.sum(np.sum(control_history**2, axis=1))
        
        # Smoothness (sum of squared accelerations)
        smoothness = self._compute_smoothness(state_history)
        
        # Settling time and overshoot
        settling_time, overshoot = self._compute_settling_metrics(
            state_history[:, 0], reference_history[:, 0]
        )
        
        # Success rate
        final_position_error = np.linalg.norm(position_error[-1])
        final_velocity_error = np.linalg.norm(velocity_error[-1])
        success_rate = 1.0 if (
            final_position_error < self.position_tolerance and
            final_velocity_error < self.velocity_tolerance
        ) else 0.0
        
        # Average computation time
        avg_computation_time = np.mean(computation_times) if computation_times else 0.0
        
        return EvaluationMetrics(
            rmse_position=rmse_position,
            rmse_velocity=rmse_velocity,
            rmse_control=rmse_control,
            control_effort=control_effort,
            smoothness=smoothness,
            settling_time=settling_time,
            overshoot=overshoot,
            computation_time=avg_computation_time,
            success_rate=success_rate,
            constraint_violations=constraint_violations
        )
        
    def _compute_smoothness(self, trajectory: np.ndarray) -> float:
        """Compute trajectory smoothness."""
        if trajectory.shape[0] < 3:
            return 0.0
            
        # Compute second differences (approximate acceleration)
        acc = np.diff(trajectory, n=2, axis=0)
        smoothness = np.sum(np.sum(acc**2, axis=1))
        
        return smoothness
        
    def _compute_settling_metrics(
        self,
        actual: np.ndarray,
        reference: np.ndarray
    ) -> Tuple[float, float]:
        """Compute settling time and overshoot."""
        error = np.abs(actual - reference)
        
        # Settling time (time to reach within 2% of final value)
        final_error = error[-1]
        tolerance = 0.02 * final_error if final_error > 0 else 0.01
        
        settling_time = 0.0
        for i in range(len(error) - 1, -1, -1):
            if error[i] > tolerance:
                settling_time = i * self.dt
                break
                
        # Overshoot (maximum overshoot percentage)
        if len(reference) > 0:
            max_error = np.max(error)
            final_ref = reference[-1]
            if final_ref != 0:
                overshoot = (max_error - final_error) / abs(final_ref) * 100
            else:
                overshoot = max_error * 100
        else:
            overshoot = 0.0
            
        return settling_time, overshoot


class Leaderboard:
    """Leaderboard for comparing MPC controllers."""
    
    def __init__(self) -> None:
        """Initialize leaderboard."""
        self.results = []
        
    def add_result(
        self,
        controller_name: str,
        test_case: str,
        metrics: EvaluationMetrics,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add evaluation result to leaderboard.
        
        Args:
            controller_name: Name of the controller
            test_case: Name of the test case
            metrics: Evaluation metrics
            config: Optional configuration dictionary
        """
        result = {
            "controller": controller_name,
            "test_case": test_case,
            "rmse_position": metrics.rmse_position,
            "rmse_velocity": metrics.rmse_velocity,
            "rmse_control": metrics.rmse_control,
            "control_effort": metrics.control_effort,
            "smoothness": metrics.smoothness,
            "settling_time": metrics.settling_time,
            "overshoot": metrics.overshoot,
            "computation_time": metrics.computation_time,
            "success_rate": metrics.success_rate,
            "constraint_violations": metrics.constraint_violations,
            "config": config
        }
        
        self.results.append(result)
        
    def get_dataframe(self) -> pd.DataFrame:
        """Get leaderboard as pandas DataFrame.
        
        Returns:
            DataFrame with all results
        """
        return pd.DataFrame(self.results)
        
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics.
        
        Returns:
            DataFrame with summary statistics
        """
        df = self.get_dataframe()
        
        summary = df.groupby("controller").agg({
            "rmse_position": ["mean", "std"],
            "rmse_velocity": ["mean", "std"],
            "rmse_control": ["mean", "std"],
            "control_effort": ["mean", "std"],
            "smoothness": ["mean", "std"],
            "settling_time": ["mean", "std"],
            "overshoot": ["mean", "std"],
            "computation_time": ["mean", "std"],
            "success_rate": ["mean", "std"],
            "constraint_violations": ["sum"]
        }).round(4)
        
        return summary
        
    def print_leaderboard(self, metric: str = "rmse_position") -> None:
        """Print leaderboard sorted by specified metric.
        
        Args:
            metric: Metric to sort by
        """
        df = self.get_dataframe()
        
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in results")
            return
            
        sorted_df = df.sort_values(metric)
        
        print(f"\nLeaderboard (sorted by {metric}):")
        print("=" * 80)
        print(f"{'Controller':<20} {'Test Case':<20} {metric:<15} {'Success Rate':<12}")
        print("-" * 80)
        
        for _, row in sorted_df.iterrows():
            print(f"{row['controller']:<20} {row['test_case']:<20} "
                  f"{row[metric]:<15.4f} {row['success_rate']:<12.2f}")
                  
    def save_results(self, filename: str) -> None:
        """Save results to CSV file.
        
        Args:
            filename: Output filename
        """
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


def create_test_cases() -> List[Dict[str, Any]]:
    """Create standard test cases for MPC evaluation.
    
    Returns:
        List of test case dictionaries
    """
    from .models import TwoLinkArm, DifferentialDrive
    
    test_cases = []
    
    # Two-link arm test cases
    arm_model = TwoLinkArm()
    
    # Test case 1: Point-to-point motion
    test_cases.append({
        "name": "arm_point_to_point",
        "model": arm_model,
        "initial_state": np.array([0.0, 0.0, 0.0, 0.0]),
        "reference_trajectory": np.array([[np.pi/4, np.pi/6, 0.0, 0.0]] * 100)
    })
    
    # Test case 2: Trajectory tracking
    t = np.linspace(0, 10, 100)
    ref_traj = np.zeros((100, 4))
    ref_traj[:, 0] = np.pi/4 * np.sin(0.5 * t)  # Sinusoidal joint 1
    ref_traj[:, 1] = np.pi/6 * np.cos(0.3 * t)  # Sinusoidal joint 2
    ref_traj[:, 2] = np.pi/4 * 0.5 * np.cos(0.5 * t)  # Joint 1 velocity
    ref_traj[:, 3] = -np.pi/6 * 0.3 * np.sin(0.3 * t)  # Joint 2 velocity
    
    test_cases.append({
        "name": "arm_trajectory_tracking",
        "model": arm_model,
        "initial_state": np.array([0.0, 0.0, 0.0, 0.0]),
        "reference_trajectory": ref_traj
    })
    
    # Differential drive test cases
    diff_model = DifferentialDrive()
    
    # Test case 3: Straight line motion
    test_cases.append({
        "name": "diff_straight_line",
        "model": diff_model,
        "initial_state": np.array([0.0, 0.0, 0.0]),
        "reference_trajectory": np.array([[t*0.5, 0.0, 0.0] for t in range(100)])
    })
    
    # Test case 4: Circular motion
    t = np.linspace(0, 2*np.pi, 100)
    ref_traj = np.zeros((100, 3))
    ref_traj[:, 0] = 2.0 * np.cos(t)  # x position
    ref_traj[:, 1] = 2.0 * np.sin(t)  # y position
    ref_traj[:, 2] = t  # orientation
    
    test_cases.append({
        "name": "diff_circular_motion",
        "model": diff_model,
        "initial_state": np.array([2.0, 0.0, 0.0]),
        "reference_trajectory": ref_traj
    })
    
    return test_cases

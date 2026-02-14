"""Demo scripts for MPC applications."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import time
import warnings

from .controller import LinearMPC, NonlinearMPC
from .models import TwoLinkArm, DifferentialDrive, create_casadi_dynamics
from .evaluation import MPCEvaluator, Leaderboard, create_test_cases
from .config import load_config, ExperimentConfig
from .utils import setup_seed, device_fallback, generate_reference_trajectory


def demo_linear_mpc() -> None:
    """Demonstrate linear MPC on a simple system."""
    print("=== Linear MPC Demo ===")
    
    # Set up deterministic simulation
    setup_seed(42)
    
    # Simple linear system: double integrator
    A = np.array([[1, 1], [0, 1]])  # State transition matrix
    B = np.array([[0.5], [1]])      # Control input matrix
    
    # MPC parameters
    N = 10  # Prediction horizon
    Q = np.diag([1, 1])  # State cost matrix
    R = np.array([[0.1]])  # Control cost matrix
    
    # Create MPC controller
    mpc = LinearMPC(
        A=A, B=B,
        prediction_horizon=N,
        state_cost_matrix=Q,
        control_cost_matrix=R,
        control_limits=(-2.0, 2.0)
    )
    
    # Simulation parameters
    dt = 0.1
    T = 5.0  # Simulation time
    n_steps = int(T / dt)
    
    # Initial state
    x0 = np.array([0.0, 0.0])
    mpc.set_state(x0)
    
    # Reference trajectory (step input)
    reference = np.array([2.0, 0.0])  # Target position
    
    # Storage
    time_vec = np.linspace(0, T, n_steps)
    state_history = np.zeros((n_steps, 2))
    control_history = np.zeros((n_steps, 1))
    reference_history = np.zeros((n_steps, 2))
    
    state_history[0] = x0
    reference_history[0] = reference
    
    print("Running simulation...")
    start_time = time.time()
    
    # Simulation loop
    for k in range(1, n_steps):
        # Solve MPC
        u_opt, x_next = mpc.solve(reference_trajectory=np.array([reference]))
        
        # Store results
        state_history[k] = x_next
        control_history[k] = u_opt
        reference_history[k] = reference
        
        # Update state
        mpc.set_state(x_next)
    
    simulation_time = time.time() - start_time
    print(f"Simulation completed in {simulation_time:.3f} seconds")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Position
    axes[0].plot(time_vec, state_history[:, 0], 'b-', label='Actual', linewidth=2)
    axes[0].plot(time_vec, reference_history[:, 0], 'r--', label='Reference', linewidth=2)
    axes[0].set_ylabel('Position')
    axes[0].set_title('Linear MPC: Position Tracking')
    axes[0].legend()
    axes[0].grid(True)
    
    # Velocity
    axes[1].plot(time_vec, state_history[:, 1], 'b-', label='Actual', linewidth=2)
    axes[1].plot(time_vec, reference_history[:, 1], 'r--', label='Reference', linewidth=2)
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Linear MPC: Velocity Tracking')
    axes[1].legend()
    axes[1].grid(True)
    
    # Control input
    axes[2].plot(time_vec, control_history[:, 0], 'g-', linewidth=2)
    axes[2].set_ylabel('Control Input')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Linear MPC: Control Input')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('assets/linear_mpc_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    position_error = np.sqrt(np.mean((state_history[:, 0] - reference_history[:, 0])**2))
    velocity_error = np.sqrt(np.mean((state_history[:, 1] - reference_history[:, 1])**2))
    control_effort = np.sum(control_history**2)
    
    print(f"\nPerformance Metrics:")
    print(f"Position RMSE: {position_error:.4f}")
    print(f"Velocity RMSE: {velocity_error:.4f}")
    print(f"Control Effort: {control_effort:.4f}")


def demo_nonlinear_mpc() -> None:
    """Demonstrate nonlinear MPC on a two-link arm."""
    print("=== Nonlinear MPC Demo ===")
    
    # Set up deterministic simulation
    setup_seed(42)
    
    # Create two-link arm model
    arm_model = TwoLinkArm(l1=1.0, l2=1.0, m1=1.0, m2=1.0)
    
    # Create CasADi dynamics function
    dynamics_func = create_casadi_dynamics(arm_model)
    
    # MPC parameters
    N = 15
    Q = np.diag([2.0, 2.0, 0.5, 0.5])  # State cost matrix
    R = np.diag([0.01, 0.01])  # Control cost matrix
    
    # Create nonlinear MPC controller
    mpc = NonlinearMPC(
        dynamics_function=dynamics_func,
        prediction_horizon=N,
        state_dim=arm_model.nx,
        control_dim=arm_model.nu,
        state_cost_matrix=Q,
        control_cost_matrix=R,
        control_limits=(-30.0, 30.0),
        state_limits=(arm_model.get_state_bounds())
    )
    
    # Simulation parameters
    dt = 0.01
    T = 8.0
    n_steps = int(T / dt)
    
    # Initial state
    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # [q1, q2, q1_dot, q2_dot]
    mpc.set_state(x0)
    
    # Reference trajectory (smooth motion to target)
    target = np.array([np.pi/3, np.pi/4, 0.0, 0.0])
    reference_traj = generate_reference_trajectory(
        start=x0, goal=target, duration=T, dt=dt, trajectory_type="smooth"
    )
    
    # Storage
    time_vec = np.linspace(0, T, n_steps)
    state_history = np.zeros((n_steps, arm_model.nx))
    control_history = np.zeros((n_steps, arm_model.nu))
    
    state_history[0] = x0
    
    print("Running nonlinear MPC simulation...")
    start_time = time.time()
    
    # Simulation loop
    for k in range(1, n_steps):
        # Get reference for current time step
        ref_idx = min(k, len(reference_traj) - 1)
        current_ref = reference_traj[ref_idx]
        
        # Create reference trajectory for MPC horizon
        ref_horizon = np.zeros((N, arm_model.nx))
        for i in range(N):
            ref_idx_horizon = min(ref_idx + i, len(reference_traj) - 1)
            ref_horizon[i] = reference_traj[ref_idx_horizon]
        
        # Solve MPC
        try:
            u_opt, x_next = mpc.solve(reference_trajectory=ref_horizon)
        except Exception as e:
            warnings.warn(f"MPC solve failed at step {k}: {e}")
            u_opt = np.zeros(arm_model.nu)
            x_next = state_history[k-1]
        
        # Store results
        state_history[k] = x_next
        control_history[k] = u_opt
        
        # Update state
        mpc.set_state(x_next)
    
    simulation_time = time.time() - start_time
    print(f"Simulation completed in {simulation_time:.3f} seconds")
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Joint angles
    axes[0, 0].plot(time_vec, state_history[:, 0], 'b-', label='Actual', linewidth=2)
    axes[0, 0].plot(time_vec, reference_traj[:, 0], 'r--', label='Reference', linewidth=2)
    axes[0, 0].set_ylabel('Joint 1 Angle (rad)')
    axes[0, 0].set_title('Nonlinear MPC: Joint 1 Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time_vec, state_history[:, 1], 'b-', label='Actual', linewidth=2)
    axes[0, 1].plot(time_vec, reference_traj[:, 1], 'r--', label='Reference', linewidth=2)
    axes[0, 1].set_ylabel('Joint 2 Angle (rad)')
    axes[0, 1].set_title('Nonlinear MPC: Joint 2 Tracking')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Joint velocities
    axes[1, 0].plot(time_vec, state_history[:, 2], 'b-', label='Actual', linewidth=2)
    axes[1, 0].plot(time_vec, reference_traj[:, 2], 'r--', label='Reference', linewidth=2)
    axes[1, 0].set_ylabel('Joint 1 Velocity (rad/s)')
    axes[1, 0].set_title('Nonlinear MPC: Joint 1 Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time_vec, state_history[:, 3], 'b-', label='Actual', linewidth=2)
    axes[1, 1].plot(time_vec, reference_traj[:, 3], 'r--', label='Reference', linewidth=2)
    axes[1, 1].set_ylabel('Joint 2 Velocity (rad/s)')
    axes[1, 1].set_title('Nonlinear MPC: Joint 2 Velocity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Control inputs
    axes[2, 0].plot(time_vec, control_history[:, 0], 'g-', linewidth=2)
    axes[2, 0].set_ylabel('Joint 1 Torque (Nm)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Nonlinear MPC: Joint 1 Control')
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(time_vec, control_history[:, 1], 'g-', linewidth=2)
    axes[2, 1].set_ylabel('Joint 2 Torque (Nm)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title('Nonlinear MPC: Joint 2 Control')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('assets/nonlinear_mpc_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    position_error = np.sqrt(np.mean(np.sum((state_history[:, :2] - reference_traj[:, :2])**2, axis=1)))
    velocity_error = np.sqrt(np.mean(np.sum((state_history[:, 2:] - reference_traj[:, 2:])**2, axis=1)))
    control_effort = np.sum(np.sum(control_history**2, axis=1))
    
    print(f"\nPerformance Metrics:")
    print(f"Position RMSE: {position_error:.4f}")
    print(f"Velocity RMSE: {velocity_error:.4f}")
    print(f"Control Effort: {control_effort:.4f}")


def demo_evaluation_framework() -> None:
    """Demonstrate the evaluation framework."""
    print("=== MPC Evaluation Framework Demo ===")
    
    # Set up deterministic simulation
    setup_seed(42)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Create evaluator
    evaluator = MPCEvaluator(
        simulation_time=5.0,
        dt=0.01,
        position_tolerance=0.1,
        velocity_tolerance=0.1
    )
    
    # Create leaderboard
    leaderboard = Leaderboard()
    
    # Test different MPC configurations
    configs = {
        "Linear MPC (N=5)": load_config("linear_mpc"),
        "Linear MPC (N=10)": load_config("linear_mpc"),
        "Nonlinear MPC": load_config("nonlinear_mpc")
    }
    
    # Modify configurations for comparison
    configs["Linear MPC (N=5)"].mpc.prediction_horizon = 5
    configs["Linear MPC (N=10)"].mpc.prediction_horizon = 10
    
    for config_name, config in configs.items():
        print(f"\nEvaluating {config_name}...")
        
        # Create controller based on config
        if "Linear" in config_name:
            # Create linear MPC
            A = np.array([[1, 1], [0, 1]])
            B = np.array([[0.5], [1]])
            
            controller = LinearMPC(
                A=A, B=B,
                prediction_horizon=config.mpc.prediction_horizon,
                state_cost_matrix=np.array(config.mpc.state_cost_matrix),
                control_cost_matrix=np.array(config.mpc.control_cost_matrix),
                control_limits=tuple(config.mpc.control_limits) if config.mpc.control_limits else None
            )
        else:
            # Create nonlinear MPC
            arm_model = TwoLinkArm(**config.model.parameters)
            dynamics_func = create_casadi_dynamics(arm_model)
            
            controller = NonlinearMPC(
                dynamics_function=dynamics_func,
                prediction_horizon=config.mpc.prediction_horizon,
                state_dim=arm_model.nx,
                control_dim=arm_model.nu,
                state_cost_matrix=np.array(config.mpc.state_cost_matrix),
                control_cost_matrix=np.array(config.mpc.control_cost_matrix),
                control_limits=tuple(config.mpc.control_limits) if config.mpc.control_limits else None,
                state_limits=tuple(map(np.array, config.mpc.state_limits)) if config.mpc.state_limits else None
            )
        
        # Evaluate on test cases
        results = evaluator.evaluate(controller, test_cases[:2])  # Use first 2 test cases
        
        # Add results to leaderboard
        for test_name, metrics in results.items():
            leaderboard.add_result(
                controller_name=config_name,
                test_case=test_name,
                metrics=metrics,
                config=config.to_dict()
            )
    
    # Print leaderboard
    leaderboard.print_leaderboard("rmse_position")
    
    # Save results
    leaderboard.save_results("assets/evaluation_results.csv")
    
    # Create comparison plots
    df = leaderboard.get_dataframe()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position RMSE comparison
    pivot_pos = df.pivot(index='test_case', columns='controller', values='rmse_position')
    pivot_pos.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Position RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].legend(title='Controller')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Control effort comparison
    pivot_effort = df.pivot(index='test_case', columns='controller', values='control_effort')
    pivot_effort.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Control Effort Comparison')
    axes[0, 1].set_ylabel('Control Effort')
    axes[0, 1].legend(title='Controller')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Computation time comparison
    pivot_time = df.pivot(index='test_case', columns='controller', values='computation_time')
    pivot_time.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Computation Time Comparison')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].legend(title='Controller')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    pivot_success = df.pivot(index='test_case', columns='controller', values='success_rate')
    pivot_success.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Success Rate Comparison')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].legend(title='Controller')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('assets/evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEvaluation completed! Results saved to assets/")


def run_all_demos() -> None:
    """Run all demonstration scripts."""
    print("Running MPC Demonstration Suite")
    print("=" * 50)
    
    try:
        demo_linear_mpc()
        print("\n" + "=" * 50)
        
        demo_nonlinear_mpc()
        print("\n" + "=" * 50)
        
        demo_evaluation_framework()
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()

#!/usr/bin/env python3
"""Simple MPC example - modernized version of the original 0654.py

This script demonstrates basic Model Predictive Control using the new MPC framework.
It replaces the original implementation with a clean, modern approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpc import LinearMPC, setup_seed, device_fallback


def main():
    """Run the basic MPC demonstration."""
    print("Model Predictive Control Demo")
    print("=" * 40)
    
    # Set up deterministic simulation
    setup_seed(42)
    device = device_fallback()
    print(f"Using device: {device}")
    
    # Define the system model (double integrator)
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
        control_limits=(-2.0, 2.0),
        device=device
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
    
    # Storage for results
    time_vec = np.linspace(0, T, n_steps)
    state_history = np.zeros((n_steps, 2))
    control_history = np.zeros((n_steps, 1))
    reference_history = np.zeros((n_steps, 2))
    
    state_history[0] = x0
    reference_history[0] = reference
    
    print("Running MPC simulation...")
    
    # Simulation loop
    for k in range(1, n_steps):
        # Solve MPC optimization
        u_opt, x_next = mpc.solve(reference_trajectory=np.array([reference]))
        
        # Store results
        state_history[k] = x_next
        control_history[k] = u_opt
        reference_history[k] = reference
        
        # Update state
        mpc.set_state(x_next)
    
    print("Simulation completed!")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Position tracking
    plt.subplot(3, 1, 1)
    plt.plot(time_vec, state_history[:, 0], 'b-', label='Actual Position', linewidth=2)
    plt.plot(time_vec, reference_history[:, 0], 'r--', label='Reference', linewidth=2)
    plt.title('MPC: Position Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    
    # Velocity tracking
    plt.subplot(3, 1, 2)
    plt.plot(time_vec, state_history[:, 1], 'b-', label='Actual Velocity', linewidth=2)
    plt.plot(time_vec, reference_history[:, 1], 'r--', label='Reference', linewidth=2)
    plt.title('MPC: Velocity Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Control input
    plt.subplot(3, 1, 3)
    plt.plot(time_vec, control_history[:, 0], 'g-', linewidth=2)
    plt.title('MPC: Control Input')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input (u)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('assets/mpc_demo_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to assets/mpc_demo_results.png")
    
    # Show plot
    plt.show()
    
    # Print performance metrics
    position_error = np.sqrt(np.mean((state_history[:, 0] - reference_history[:, 0])**2))
    velocity_error = np.sqrt(np.mean((state_history[:, 1] - reference_history[:, 1])**2))
    control_effort = np.sum(control_history**2)
    
    print(f"\nPerformance Metrics:")
    print(f"Position RMSE: {position_error:.4f}")
    print(f"Velocity RMSE: {velocity_error:.4f}")
    print(f"Control Effort: {control_effort:.4f}")
    
    print(f"\nFinal State: [{state_history[-1, 0]:.3f}, {state_history[-1, 1]:.3f}]")
    print(f"Target State: [{reference[0]:.3f}, {reference[1]:.3f}]")


if __name__ == "__main__":
    main()
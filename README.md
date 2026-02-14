# Model Predictive Control for Robotics

A comprehensive MPC framework for robotics applications with support for linear and nonlinear systems, advanced constraints, and evaluation metrics.

## Features

- **Linear MPC**: Fast optimization for linear systems using quadratic programming
- **Nonlinear MPC**: Advanced control for nonlinear systems using CasADi optimization
- **Robot Models**: Built-in models for two-link arms and differential drive robots
- **Constraints**: Support for state and control constraints
- **Evaluation Framework**: Comprehensive metrics and leaderboard system
- **Visualization**: Interactive demos and plotting tools
- **Configuration Management**: YAML-based configuration system

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Core dependencies
pip install numpy scipy matplotlib casadi torch pandas pyyaml

# Optional dependencies for advanced features
pip install plotly streamlit gradio  # For interactive demos
pip install pytest black ruff mypy   # For development
```

### Install Package

```bash
# Development installation
pip install -e .

# Or install from source
git clone https://github.com/kryptologyst/Model-Predictive-Control-for-Robotics.git
cd Model-Predictive-Control-for-Robotics
pip install -e .
```

## Quick Start

### Basic Linear MPC

```python
import numpy as np
from mpc import LinearMPC

# Define linear system: x(k+1) = A*x(k) + B*u(k)
A = np.array([[1, 1], [0, 1]])  # State transition matrix
B = np.array([[0.5], [1]])      # Control input matrix

# Create MPC controller
mpc = LinearMPC(
    A=A, B=B,
    prediction_horizon=10,
    state_cost_matrix=np.diag([1, 1]),
    control_cost_matrix=np.array([[0.1]]),
    control_limits=(-2.0, 2.0)
)

# Set initial state
mpc.set_state(np.array([0.0, 0.0]))

# Solve for control input
control_input, next_state = mpc.solve()
```

### Nonlinear MPC for Robot Arm

```python
from mpc import NonlinearMPC, TwoLinkArm, create_casadi_dynamics

# Create robot model
arm_model = TwoLinkArm(l1=1.0, l2=1.0, m1=1.0, m2=1.0)

# Create CasADi dynamics function
dynamics_func = create_casadi_dynamics(arm_model)

# Create nonlinear MPC controller
mpc = NonlinearMPC(
    dynamics_function=dynamics_func,
    prediction_horizon=15,
    state_dim=arm_model.nx,
    control_dim=arm_model.nu,
    state_cost_matrix=np.diag([2.0, 2.0, 0.5, 0.5]),
    control_cost_matrix=np.diag([0.01, 0.01]),
    control_limits=(-30.0, 30.0)
)

# Set initial state
mpc.set_state(np.array([0.0, 0.0, 0.0, 0.0]))

# Solve for control input
control_input, next_state = mpc.solve()
```

### Running Demos

```bash
# Run all demonstrations
python -m mpc.demos

# Or use the command line tool
mpc-demo
```

## Project Structure

```
mpc-robotics/
├── src/mpc/                    # Main package
│   ├── __init__.py
│   ├── controller.py           # MPC controller implementations
│   ├── models.py              # Robot models and dynamics
│   ├── evaluation.py         # Evaluation framework
│   ├── config.py             # Configuration management
│   ├── utils.py              # Utility functions
│   └── demos.py              # Demonstration scripts
├── configs/                   # Configuration files
├── tests/                     # Unit tests
├── assets/                    # Generated plots and results
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Configuration

The framework uses YAML-based configuration files for easy experimentation:

```yaml
# configs/linear_mpc.yaml
mpc:
  prediction_horizon: 10
  state_cost_matrix: [[1.0, 0.0], [0.0, 1.0]]
  control_cost_matrix: [[0.1]]
  control_limits: [-2.0, 2.0]

model:
  model_type: "TwoLinkArm"
  parameters:
    l1: 1.0
    l2: 1.0
    m1: 1.0
    m2: 1.0

evaluation:
  simulation_time: 10.0
  dt: 0.01
  position_tolerance: 0.1
  velocity_tolerance: 0.1
```

## Evaluation Framework

The framework includes a comprehensive evaluation system with metrics:

- **Tracking Performance**: RMSE for position and velocity
- **Control Quality**: Control effort and smoothness
- **Temporal Performance**: Settling time and overshoot
- **Computational Performance**: Solution time and success rate
- **Constraint Handling**: Violation detection and counting

```python
from mpc import MPCEvaluator, Leaderboard, create_test_cases

# Create evaluator
evaluator = MPCEvaluator(simulation_time=10.0, dt=0.01)

# Create test cases
test_cases = create_test_cases()

# Evaluate controller
results = evaluator.evaluate(mpc_controller, test_cases)

# Create leaderboard
leaderboard = Leaderboard()
for test_name, metrics in results.items():
    leaderboard.add_result("MyController", test_name, metrics)

# Print results
leaderboard.print_leaderboard("rmse_position")
```

## Robot Models

### Two-Link Arm

A planar two-link robotic arm with configurable parameters:

```python
from mpc import TwoLinkArm

arm = TwoLinkArm(
    l1=1.0,  # Length of first link
    l2=1.0,  # Length of second link
    m1=1.0,  # Mass of first link
    m2=1.0,  # Mass of second link
    g=9.81   # Gravitational acceleration
)

# State: [q1, q2, q1_dot, q2_dot]
# Control: [tau1, tau2]
```

### Differential Drive Robot

A mobile robot with differential drive:

```python
from mpc import DifferentialDrive

robot = DifferentialDrive(
    wheel_base=0.5,    # Distance between wheels
    wheel_radius=0.1   # Wheel radius
)

# State: [x, y, theta]
# Control: [v_left, v_right]
```

## Advanced Features

### Constraints

Both linear and nonlinear MPC support state and control constraints:

```python
# Control constraints
mpc = LinearMPC(
    # ... other parameters ...
    control_limits=(-5.0, 5.0)  # Min/max control input
)

# State constraints
mpc = NonlinearMPC(
    # ... other parameters ...
    state_limits=(x_min, x_max)  # Min/max state bounds
)
```

### Warm Starts

The nonlinear MPC implementation includes warm starts for faster convergence:

```python
# Previous solution is automatically used as initial guess
u_opt, x_next = mpc.solve()
```

### Device Support

Automatic device detection for GPU acceleration:

```python
from mpc import device_fallback

device = device_fallback()  # Returns 'cuda', 'mps', or 'cpu'
mpc = LinearMPC(..., device=device)
```

## Safety and Limitations

### Important Disclaimers

**DO NOT USE ON REAL ROBOTS WITHOUT EXPERT REVIEW**

This software is designed for research and educational purposes only. Real-world deployment requires:

1. **Safety Certification**: Professional safety analysis and certification
2. **Hardware Testing**: Extensive testing on actual hardware
3. **Fail-Safe Mechanisms**: Emergency stop systems and safety interlocks
4. **Expert Review**: Validation by robotics and control experts
5. **Risk Assessment**: Comprehensive risk analysis for the specific application

### Known Limitations

- Simplified dynamics models (no friction, backlash, or flexibility)
- No real-time guarantees for optimization
- Limited to simulation environments
- No hardware interface implementations
- Basic constraint handling (no soft constraints)

### Safety Features

- Control input limits and saturation
- State bounds checking
- Graceful failure handling
- Deterministic seeding for reproducibility
- Clear error messages and warnings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Format code: `black src/` and `ruff check src/`
7. Submit a pull request

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mpc

# Run specific test file
pytest tests/test_controller.py
```

## Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Format code
black src/
ruff check src/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{mpc_robotics_2026,
  title={Model Predictive Control for Robotics},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Model-Predictive-Control-for-Robotics}
}
```

## Acknowledgments

- CasADi optimization framework
- NumPy and SciPy scientific computing
- PyTorch for tensor operations
- The robotics and control community

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation in `docs/`
- Review the example scripts in `src/mpc/demos.py`

---

**Remember**: This is research/educational software. Always consult robotics experts before deploying on real hardware.
# Model-Predictive-Control-for-Robotics

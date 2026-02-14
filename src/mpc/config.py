"""Configuration management for MPC applications."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
import numpy as np
from pathlib import Path


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    prediction_horizon: int = 10
    state_cost_matrix: List[List[float]] = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    control_cost_matrix: List[List[float]] = field(default_factory=lambda: [[0.1]])
    control_limits: Optional[List[float]] = None
    state_limits: Optional[List[List[float]]] = None
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_horizon": self.prediction_horizon,
            "state_cost_matrix": self.state_cost_matrix,
            "control_cost_matrix": self.control_cost_matrix,
            "control_limits": self.control_limits,
            "state_limits": self.state_limits,
            "device": self.device
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MPCConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelConfig:
    """Configuration for robot models."""
    model_type: str = "TwoLinkArm"
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    simulation_time: float = 10.0
    dt: float = 0.01
    position_tolerance: float = 0.1
    velocity_tolerance: float = 0.1
    test_cases: List[str] = field(default_factory=lambda: ["arm_point_to_point", "arm_trajectory_tracking"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "simulation_time": self.simulation_time,
            "dt": self.dt,
            "position_tolerance": self.position_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
            "test_cases": self.test_cases
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    mpc: MPCConfig = field(default_factory=MPCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment_name: str = "mpc_experiment"
    seed: int = 42
    output_dir: str = "results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mpc": self.mpc.to_dict(),
            "model": self.model.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "output_dir": self.output_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            mpc=MPCConfig.from_dict(data.get("mpc", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            evaluation=EvaluationConfig.from_dict(data.get("evaluation", {})),
            experiment_name=data.get("experiment_name", "mpc_experiment"),
            seed=data.get("seed", 42),
            output_dir=data.get("output_dir", "results")
        )
    
    def save(self, filename: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filename: str) -> "ExperimentConfig":
        """Load configuration from YAML file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded configuration
        """
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create default configurations for different scenarios.
    
    Returns:
        Dictionary of configurations
    """
    configs = {}
    
    # Basic linear MPC
    configs["linear_mpc"] = ExperimentConfig(
        mpc=MPCConfig(
            prediction_horizon=10,
            state_cost_matrix=[[1.0, 0.0], [0.0, 1.0]],
            control_cost_matrix=[[0.1]],
            control_limits=[-5.0, 5.0]
        ),
        model=ModelConfig(
            model_type="TwoLinkArm",
            parameters={"l1": 1.0, "l2": 1.0, "m1": 1.0, "m2": 1.0}
        ),
        evaluation=EvaluationConfig(
            simulation_time=10.0,
            dt=0.01,
            test_cases=["arm_point_to_point"]
        ),
        experiment_name="linear_mpc_basic"
    )
    
    # Advanced nonlinear MPC
    configs["nonlinear_mpc"] = ExperimentConfig(
        mpc=MPCConfig(
            prediction_horizon=15,
            state_cost_matrix=[[2.0, 0.0, 0.0, 0.0], 
                             [0.0, 2.0, 0.0, 0.0],
                             [0.0, 0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.0, 0.5]],
            control_cost_matrix=[[0.01, 0.0], [0.0, 0.01]],
            control_limits=[-50.0, 50.0],
            state_limits=[[-np.pi, -np.pi, -10.0, -10.0],
                        [np.pi, np.pi, 10.0, 10.0]]
        ),
        model=ModelConfig(
            model_type="TwoLinkArm",
            parameters={"l1": 1.0, "l2": 1.0, "m1": 1.0, "m2": 1.0}
        ),
        evaluation=EvaluationConfig(
            simulation_time=15.0,
            dt=0.01,
            test_cases=["arm_point_to_point", "arm_trajectory_tracking"]
        ),
        experiment_name="nonlinear_mpc_advanced"
    )
    
    # Differential drive MPC
    configs["differential_drive"] = ExperimentConfig(
        mpc=MPCConfig(
            prediction_horizon=20,
            state_cost_matrix=[[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.1]],
            control_cost_matrix=[[0.1, 0.0], [0.0, 0.1]],
            control_limits=[-3.0, 3.0]
        ),
        model=ModelConfig(
            model_type="DifferentialDrive",
            parameters={"wheel_base": 0.5, "wheel_radius": 0.1}
        ),
        evaluation=EvaluationConfig(
            simulation_time=20.0,
            dt=0.01,
            test_cases=["diff_straight_line", "diff_circular_motion"]
        ),
        experiment_name="differential_drive_mpc"
    )
    
    return configs


def load_config(config_name: str) -> ExperimentConfig:
    """Load configuration by name.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration object
    """
    configs = create_default_configs()
    
    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        
    return configs[config_name]

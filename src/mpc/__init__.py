"""Model Predictive Control (MPC) package for robotics applications."""

from .controller import MPCController, LinearMPC, NonlinearMPC
from .models import LinearSystem, NonlinearSystem, RobotModel
from .utils import setup_seed, device_fallback

__version__ = "1.0.0"
__all__ = [
    "MPCController",
    "LinearMPC", 
    "NonlinearMPC",
    "LinearSystem",
    "NonlinearSystem", 
    "RobotModel",
    "setup_seed",
    "device_fallback",
]

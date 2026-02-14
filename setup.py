"""Model Predictive Control (MPC) for Robotics Applications

A comprehensive MPC framework for robotics with support for linear and nonlinear systems,
advanced constraints, and evaluation metrics.

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Model Predictive Control (MPC) for Robotics Applications"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="mpc-robotics",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="Model Predictive Control (MPC) for Robotics Applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/mpc-robotics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "streamlit>=1.25.0",
            "gradio>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mpc-demo=mpc.demos:run_all_demos",
        ],
    },
    include_package_data=True,
    package_data={
        "mpc": ["configs/*.yaml", "assets/*"],
    },
    zip_safe=False,
)

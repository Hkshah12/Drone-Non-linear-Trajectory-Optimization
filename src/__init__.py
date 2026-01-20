"""
Drone Racing Path Planning Package

A comprehensive implementation of RRT-based path planning and trajectory
optimization for autonomous drone racing through hoops while avoiding obstacles.

Modules:
    - rrt: RRT algorithms for path planning
    - drone_dynamics: Drone physics and motion models
    - trajectory_optimization: Trajectory optimization functions
    - helpers_obstacles: Helper functions and obstacle classes
"""

import numpy as np

# Shared random generator for reproducibility across all modules
# Using a fixed seed ensures consistent results for testing and debugging
rng = np.random.default_rng(12345)

# Package version
__version__ = "1.0.0"

# Package author
__author__ = "Your Name"

# Expose key classes and functions at package level
from .rrt import (
    generate_random_point,
    distance_euclidean,
    find_nearest_node,
    steer_naive,
    run_rrt,
    get_rrt_path,
    drone_racing_rrt,
    run_rrt_with_obstacles,
    drone_racing_rrt_with_obstacles
)

from .drone_dynamics import (
    compute_attitude_from_ypr,
    compute_force,
    compute_terminal_velocity,
    generate_random_pose,
    find_nearest_pose,
    steer_with_terminal_velocity,
    compute_force_with_gravity,
    steer
)

from .trajectory_optimization import (
    angle_diff,
    pack_decision_vars,
    unpack_decision_vars,
    cost_function_thrust,
    cost_function_angular,
    cost_function_smoothness,
    cost_function_gimbal_lock,
    cost_function_tuned,
    dynamics_constraints_robust,
    boundary_constraints_robust,
    collision_constraints_optimized,
    initialize_from_rrt_robust
)

__all__ = [
    # RRT
    'generate_random_point',
    'distance_euclidean',
    'find_nearest_node',
    'steer_naive',
    'run_rrt',
    'get_rrt_path',
    'drone_racing_rrt',
    'run_rrt_with_obstacles',
    'drone_racing_rrt_with_obstacles',
    # Drone dynamics
    'compute_attitude_from_ypr',
    'compute_force',
    'compute_terminal_velocity',
    'generate_random_pose',
    'find_nearest_pose',
    'steer_with_terminal_velocity',
    'compute_force_with_gravity',
    'steer',
    # Trajectory optimization
    'angle_diff',
    'pack_decision_vars',
    'unpack_decision_vars',
    'cost_function_thrust',
    'cost_function_angular',
    'cost_function_smoothness',
    'cost_function_gimbal_lock',
    'cost_function_tuned',
    'dynamics_constraints_robust',
    'boundary_constraints_robust',
    'collision_constraints_optimized',
    'initialize_from_rrt_robust',
    # Shared
    'rng',
]
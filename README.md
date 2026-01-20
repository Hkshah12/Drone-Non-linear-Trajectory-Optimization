🎯 Overview
This project implements a complete autonomous drone racing pipeline:

RRT Path Planning: Rapidly-exploring Random Trees for collision-free path generation
Drone Dynamics: Physics-based motion model with terminal velocity constraints
Obstacle Avoidance: Sphere-based collision detection and avoidance
Trajectory Optimization: Smooth, dynamically-feasible trajectory generation using nonlinear optimization

The system plans paths through racing hoops while avoiding obstacles, then optimizes the trajectory for smooth, efficient flight.
Pipeline Overview
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
│ Start/Goal  │ ──▶ │ RRT Planning │ ──▶ │ Trajectory          │ ──▶ │ Optimized    │
│ + Hoops     │     │ (Collision-  │     │ Optimization        │     │ Smooth Path  │
│ + Obstacles │     │  Free Path)  │     │ (Physics + Costs)   │     │              │
└─────────────┘     └──────────────┘     └─────────────────────┘     └──────────────┘

✨ Features

RRT Path Planning

Basic 3D RRT with Point3 configurations
Pose-based RRT with full 6-DOF (position + orientation)
Goal-biased sampling for faster convergence
Vectorized nearest-neighbor search


Drone Dynamics

Terminal velocity motion model
Attitude computation from Euler angles (yaw, pitch, roll)
Force and thrust calculations with gravity
Realistic steering with control limits


Obstacle Avoidance

Sphere obstacle collision detection
Segment-based collision checking
Safety margin enforcement
Support for easy/hard obstacle configurations


Trajectory Optimization

Cost functions: thrust deviation, angular velocity, smoothness, gimbal lock
Dynamics constraints (physics consistency)
Boundary constraints (start, goal, hoops)
Collision constraints (obstacle avoidance)
RRT-based initialization for optimization


Visualization

3D interactive plots with Plotly
RRT tree visualization
Path comparison (RRT vs optimized)
Racing course with hoops and obstacles




📁 Project Structure
drone-racing-path-planning/
│
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization + shared RNG
│   ├── helpers_obstacles.py          # Helper functions (visualization, obstacles)
│   ├── rrt.py                        # RRT algorithms (TODO 1-5, 14-16)
│   ├── drone_dynamics.py             # Drone physics (TODO 6-13)
│   └── trajectory_optimization.py    # Optimization functions (TODO 17-28)
│
├── tests/                            # Unit tests
│   └── main.py                       # All test cases
│
├── notebooks/                        # Original notebooks (optional)
│   └── original_notebook.ipynb
│
├── main.py                           # Entry point with demos
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules

🚀 Installation
Prerequisites

Python 3.9 or higher
pip (Python package manager)
Git

Step-by-Step Installation
1. Clone the Repository
bashgit clone https://github.com/yourusername/drone-racing-path-planning.git
cd drone-racing-path-planning
2. Create a Virtual Environment (Recommended)
Windows (PowerShell):
powershellpython -m venv venv
.\venv\Scripts\Activate.ps1
Windows (Command Prompt):
cmdpython -m venv venv
venv\Scripts\activate.bat
macOS/Linux:
bashpython3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bashpip install -r requirements.txt
4. Verify Installation
bashpython -c "import gtsam; import numpy; print('✅ Installation successful!')"
Troubleshooting Installation
If pip install fails due to spaces in path:
bashpython -m pip install -r requirements.txt
If GTSAM installation fails:
bash# Try conda instead
conda install -c conda-forge gtsam

⚡ Quick Start
Run All Demos
bashpython main.py
This will run through all demonstrations:

Basic 3D RRT
RRT with drone dynamics
Simple trajectory optimization
Racing through hoops
Racing with obstacles
Hard obstacles preview

Run Specific Demo
pythonfrom main import demo_racing_with_obstacles

# Run the final challenge demo
demo_racing_with_obstacles()
Run Tests
bashpython -m pytest tests/main.py -v

📖 Usage
Basic RRT Path Planning
pythonimport gtsam
from src.rrt import (
    generate_random_point,
    distance_euclidean,
    find_nearest_node,
    steer_naive,
    run_rrt,
    get_rrt_path
)

# Define start and goal
start = gtsam.Point3(1, 2, 3)
goal = gtsam.Point3(8, 8, 8)

# Run RRT
rrt_tree, parents = run_rrt(
    start, goal,
    generate_random_point,
    steer_naive,
    distance_euclidean,
    find_nearest_node,
    threshold=0.5
)

# Extract path
path = get_rrt_path(rrt_tree, parents)
print(f"Path found with {len(path)} waypoints")
RRT with Drone Dynamics
pythonimport gtsam
from src.rrt import run_rrt, get_rrt_path
from src.drone_dynamics import (
    generate_random_pose,
    find_nearest_pose,
    steer
)
from src import helpers_obstacles as helpers

# Define start and goal poses
start = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
goal = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 8))

# Run RRT with drone dynamics
rrt_tree, parents = run_rrt(
    start, goal,
    generate_random_pose,
    steer,
    helpers.distance_between_poses,
    find_nearest_pose,
    threshold=1.5
)

path = get_rrt_path(rrt_tree, parents)
Racing Through Hoops
pythonimport gtsam
from src.rrt import drone_racing_rrt
from src import helpers_obstacles as helpers

# Setup
start = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))
targets = helpers.get_targets()
hoops = helpers.get_hoops()

# Plan path through all hoops
racing_path = drone_racing_rrt(start, targets)
print(f"Racing path: {len(racing_path)} waypoints")

# Visualize
helpers.drone_racing_path(hoops, start, racing_path)
Racing with Obstacles
pythonfrom src.rrt import drone_racing_rrt_with_obstacles
from src import helpers_obstacles as helpers

# Get obstacles
obstacles = helpers.get_obstacles_easy()  # or get_obstacles_hard()

# Plan collision-free path
path = drone_racing_rrt_with_obstacles(start, targets, obstacles)

# Verify collision-free
has_collision, _ = helpers.check_path_collision(path, obstacles)
print(f"Collision-free: {not has_collision}")

# Visualize
helpers.drone_racing_path_with_obstacles(hoops, start, path, obstacles)
Trajectory Optimization
pythonfrom src.trajectory_optimization import (
    initialize_from_rrt_robust,
    dynamics_constraints_robust,
    boundary_constraints_robust,
    collision_constraints_optimized,
    cost_function_tuned,
    unpack_decision_vars
)
from src import helpers_obstacles as helpers

# Optimize RRT path
optimized_path, success, info = helpers.optimize_trajectory(
    rrt_path=path,
    start_pose=start,
    goal_position=goal.translation(),
    hoops=hoops,
    obstacles=obstacles,
    N=25,
    dt=0.1,
    weights={'thrust': 0.1, 'angular': 1.0, 'smoothness': 5.0},
    initialize_from_rrt_robust=initialize_from_rrt_robust,
    dynamics_constraints_robust=dynamics_constraints_robust,
    boundary_constraints_robust=boundary_constraints_robust,
    collision_constraints_optimized=collision_constraints_optimized,
    cost_function_tuned=cost_function_tuned,
    unpack_decision_vars=unpack_decision_vars
)

if success:
    print(f"Optimization successful! Cost: {info['cost']:.2f}")

📚 Module Documentation
src/rrt.py - RRT Algorithms
FunctionDescriptiongenerate_random_point(target)Generate random 3D point with goal biasdistance_euclidean(p1, p2)Compute Euclidean distancefind_nearest_node(rrt, node)Find nearest node in tree (vectorized)steer_naive(parent, target, fraction)Simple linear steeringrun_rrt(start, target, ...)Main RRT algorithmget_rrt_path(rrt, parents)Extract path from treedrone_racing_rrt(start, targets)Race through multiple hoopsrun_rrt_with_obstacles(...)RRT with collision checkingdrone_racing_rrt_with_obstacles(...)Racing with obstacle avoidance
src/drone_dynamics.py - Drone Physics
FunctionDescriptioncompute_attitude_from_ypr(y, p, r)Rotation matrix from Euler anglescompute_force(attitude, thrust)Force vector from attitude + thrustcompute_terminal_velocity(force)Terminal velocity from forcegenerate_random_pose(target)Random 6-DOF pose with goal biasfind_nearest_pose(rrt, node)Find nearest pose in treesteer_with_terminal_velocity(...)Steering with velocity modelcompute_force_with_gravity(...)Force including gravitysteer(current, target, duration)Realistic steering with control limits
src/trajectory_optimization.py - Optimization
FunctionDescriptionangle_diff(a, b)Angular difference with wrappingpack_decision_vars(states, controls, N)Pack into flat vectorunpack_decision_vars(z, N)Unpack into states/controlscost_function_thrust(...)Thrust deviation costcost_function_angular(...)Angular velocity costcost_function_smoothness(...)Control smoothness costcost_function_gimbal_lock(...)Gimbal lock penaltycost_function_tuned(z, N, weights)Combined cost functiondynamics_constraints_robust(...)Physics constraintsboundary_constraints_robust(...)Start/goal/hoop constraintscollision_constraints_optimized(...)Obstacle avoidance constraintsinitialize_from_rrt_robust(...)Initialize optimization from RRT

🧪 Running Tests
Run All Tests
bashpython -m pytest tests/main.py -v
Run Specific Test Class
bash# RRT tests
python -m pytest tests/main.py::TestRRT -v

# Drone dynamics tests
python -m pytest tests/main.py::TestDroneDynamics -v

# Cost function tests
python -m pytest tests/main.py::TestCostFunctions -v

# Constraint tests
python -m pytest tests/main.py::TestConstraints -v
Run Single Test
bashpython -m pytest tests/main.py::TestRRT::test_generate_random_point -v
Run with Coverage
bashpip install pytest-cov
python -m pytest tests/main.py --cov=src --cov-report=html

📝 Task Overview
This project implements 28 tasks organized into categories:
RRT Basics (TODO 1-5)
#TaskDescription1generate_random_pointRandom sampling with goal bias2distance_euclideanDistance metric3find_nearest_nodeNearest neighbor (vectorized)4steer_naiveSimple steering function5run_rrtMain RRT loop
Drone Dynamics (TODO 6-13)
#TaskDescription6compute_attitude_from_yprEuler to rotation matrix7compute_forceThrust to force vector8compute_terminal_velocityForce to velocity9generate_random_poseRandom 6-DOF sampling10find_nearest_poseNearest pose in tree11steer_with_terminal_velocityVelocity-based steering12compute_force_with_gravityForce with gravity13steerRealistic steering
Racing & Obstacles (TODO 14-16)
#TaskDescription14drone_racing_rrtMulti-hoop racing15run_rrt_with_obstaclesCollision-free RRT16drone_racing_rrt_with_obstaclesRacing with obstacles
Trajectory Optimization (TODO 17-28)
#TaskDescription17angle_diffAngle wrapping18pack_decision_varsPack for optimizer19unpack_decision_varsUnpack from optimizer20cost_function_thrustThrust cost21cost_function_angularAngular cost22cost_function_smoothnessSmoothness cost23cost_function_gimbal_lockGimbal lock penalty24cost_function_tunedCombined cost25dynamics_constraints_robustPhysics constraints26boundary_constraints_robustBoundary constraints27collision_constraints_optimizedCollision constraints28initialize_from_rrt_robustRRT initialization

📦 Dependencies
Core Dependencies
PackageVersionPurposenumpy≥1.25Numerical computinggtsam≥4.2Geometry & optimizationscipy≥1.10Optimization algorithmsplotly≥5.0Interactive visualizationpandas≥2.0Data handling
Development Dependencies
PackagePurposepytestTesting frameworkpytest-covCoverage reports
requirements.txt
numpy==1.25
gtsam==4.2
scipy>=1.10
plotly>=5.0
pandas>=2.0
pytest>=7.0

❓ Troubleshooting
Common Issues
1. Import Error: No module named 'src'
bash# Make sure you're in the project root directory
cd drone-racing-path-planning

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
2. GTSAM installation fails
bash# Use conda instead of pip
conda install -c conda-forge gtsam
3. Plotly figures not showing
python# Use fig.show() or save to HTML
fig.write_html("output.html")
4. "Unable to create process" error (Windows)
bash# Use python -m pip instead of pip
python -m pip install -r requirements.txt
5. Tests fail with import errors
bash# Install package in development mode
pip install -e .

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Code Style

Follow PEP 8 guidelines
Add docstrings to all functions
Include type hints
Write unit tests for new features


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

GTSAM library for geometry and optimization
Georgia Tech Robot Intelligence Lab for drone dynamics model
Course instructors and TAs for project guidance


# Drone Racing Trajectory Optimization

This project implements RRT (Rapidly-exploring Random Trees) for 3D drone path planning, accounting for drone dynamics like thrust and drag.

## Structure

- `src/`: Contains the core algorithms (`rrt.py`) and physics models (`drone_dynamics.py`).
- `main.py`: Entry point to run the simulation.
- `requirements.txt`: Python dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```

## Details

The project optimizes a trajectory for a drone to fly through a sequence of hoops or from a start to a goal position using sampling-based motion planning.

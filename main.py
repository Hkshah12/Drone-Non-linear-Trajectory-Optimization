"""Main entry point - Drone Racing Path Planning demonstrations."""

import math
import gtsam
import numpy as np

from src.rrt import (
    generate_random_point,
    distance_euclidean,
    find_nearest_node,
    steer_naive,
    run_rrt,
    get_rrt_path
)
from src.drone_dynamics import (
    generate_random_pose,
    find_nearest_pose,
    steer_with_terminal_velocity
)
from src import helpers_obstacles as helpers


# =============================================================================
# Demo 1: Basic 3D RRT with Point3
# =============================================================================
def demo_rrt_3d():
    """Demonstrate basic 3D RRT using Point3."""
    print("\n=== RRT 3D Demo ===")
    
    start_rrt_3d = gtsam.Point3(1, 2, 3)
    target_rrt_3d = gtsam.Point3(4, 7, 2)
    
    # Run RRT
    rrt_3d, parents_rrt_3d = run_rrt(
        start_rrt_3d,
        target_rrt_3d,
        generate_random_point,
        steer_naive,
        distance_euclidean,
        find_nearest_node,
        threshold=0.1
    )
    print(f"Nodes in RRT: {len(rrt_3d)}")
    
    # Visualize tree
    helpers.visualize_tree(rrt_3d, parents_rrt_3d, start_rrt_3d, target_rrt_3d)
    
    # Extract and visualize path
    path_rrt_3d = get_rrt_path(rrt_3d, parents_rrt_3d)
    print(f"Length of Path: {len(path_rrt_3d)}")
    helpers.visualize_path(path_rrt_3d, start_rrt_3d, target_rrt_3d)
    
    return rrt_3d, parents_rrt_3d, path_rrt_3d


# =============================================================================
# Demo 2: RRT with Drone Dynamics (Pose3)
# =============================================================================
def demo_rrt_drone_dynamics():
    """Demonstrate RRT with drone dynamics using Pose3."""
    print("\n=== RRT with Drone Dynamics Demo ===")
    
    start_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
    target_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(5, 7, 4))
    
    # Run RRT with drone dynamics
    rrt_drone, parents_rrt_drone = run_rrt(
        start_rrt_drone,
        target_rrt_drone,
        generate_random_pose,
        steer_with_terminal_velocity,
        helpers.distance_between_poses,
        find_nearest_pose,
        threshold=1.5
    )
    print(f"RRT Completed. Number of nodes: {len(rrt_drone)}")
    
    # Visualize tree
    helpers.visualize_tree(rrt_drone, parents_rrt_drone, start_rrt_drone, target_rrt_drone)
    
    # Extract path
    path_rrt_drone = get_rrt_path(rrt_drone, parents_rrt_drone)
    print(f"Path found with length: {len(path_rrt_drone)}")
    
    # Print first few path poses
    print("Path poses (first 5):")
    for p in path_rrt_drone[:5]:
        print(f"  {p.translation()}")
    if len(path_rrt_drone) > 5:
        print(f"  ... and {len(path_rrt_drone) - 5} more")
    
    # Animate the drone path
    print("\nAnimating path with terminal velocity steering:")
    helpers.animate_drone_path(path_rrt_drone, start_rrt_drone, target_rrt_drone)
    
    return rrt_drone, parents_rrt_drone, path_rrt_drone


# =============================================================================
# Demo 3: Drone Racing Path
# =============================================================================
def demo_drone_racing():
    """Demonstrate drone racing path visualization."""
    print("\n=== Drone Racing Demo ===")
    
    # Starting pose: Yaw 45 deg, Position (1, 3, 8)
    start_race = gtsam.Pose3(
        r=gtsam.Rot3.Yaw(math.radians(45)),
        t=gtsam.Point3(1, 3, 8)
    )
    
    # Get hoops and obstacles
    hoops = helpers.get_hoops()
    obstacles = helpers.get_obstacles_easy()
    
    # Visualize racing path (without obstacles)
    print("Racing path without obstacles:")
    helpers.drone_racing_path(hoops, start_race, [])
    
    # Visualize racing path (with obstacles)
    print("Racing path with obstacles:")
    helpers.drone_racing_path_with_obstacles(hoops, start_race, [], obstacles=obstacles)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run all demonstrations."""
    print("=" * 50)
    print("Starting Drone Racing Path Planning...")
    print("=" * 50)
    
    # Run demos
    demo_rrt_3d()
    demo_rrt_drone_dynamics()
    demo_drone_racing()
    
    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
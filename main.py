import math
import time
import gtsam
import numpy as np

from src.rrt import (
    generate_random_point,
    distance_euclidean,
    find_nearest_node,
    steer_naive,
    run_rrt,
    get_rrt_path,
    run_rrt_with_obstacles,
    drone_racing_rrt,
    drone_racing_rrt_with_obstacles
)
from src.drone_dynamics import (
    generate_random_pose,
    find_nearest_pose,
    steer_with_terminal_velocity,
    steer
)
from src.trajectory_optimization import (
    initialize_from_rrt_robust,
    dynamics_constraints_robust,
    boundary_constraints_robust,
    collision_constraints_optimized,
    cost_function_tuned,
    unpack_decision_vars
)
from src import helpers_obstacles as helpers


def compute_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        pos1 = np.array(path[i].translation())
        pos2 = np.array(path[i + 1].translation())
        length += np.linalg.norm(pos2 - pos1)
    return length


def demo_rrt_3d():
    print("\n" + "=" * 60)
    print("Demo 1: Basic RRT 3D")
    print("=" * 60)

    start_rrt_3d = gtsam.Point3(1, 2, 3)
    target_rrt_3d = gtsam.Point3(4, 7, 2)

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

    helpers.visualize_tree(rrt_3d, parents_rrt_3d, start_rrt_3d, target_rrt_3d)

    path_rrt_3d = get_rrt_path(rrt_3d, parents_rrt_3d)
    print(f"Length of Path: {len(path_rrt_3d)}")
    helpers.visualize_path(path_rrt_3d, start_rrt_3d, target_rrt_3d)

    return rrt_3d, parents_rrt_3d, path_rrt_3d


def demo_rrt_drone_dynamics():
    print("\n" + "=" * 60)
    print("Demo 2: RRT with Drone Dynamics")
    print("=" * 60)

    start_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
    target_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(5, 7, 4))

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

    helpers.visualize_tree(rrt_drone, parents_rrt_drone, start_rrt_drone, target_rrt_drone)

    path_rrt_drone = get_rrt_path(rrt_drone, parents_rrt_drone)
    print(f"Length of Path: {len(path_rrt_drone)}")

    print("Animating path with terminal velocity steering:")
    helpers.animate_drone_path(path_rrt_drone, start_rrt_drone, target_rrt_drone)

    return rrt_drone, parents_rrt_drone, path_rrt_drone


def demo_simple_optimization():
    print("\n" + "=" * 60)
    print("Demo 3: Simple Trajectory Optimization")
    print("=" * 60)

    start_simple = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 5))
    goal_simple = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 5))

    print("Step 1: Running RRT...")
    rrt_simple, parents_simple = run_rrt(
        start_simple, goal_simple,
        generate_random_pose, steer, helpers.distance_between_poses,
        find_nearest_pose, threshold=2.0
    )
    path_simple = get_rrt_path(rrt_simple, parents_simple)
    print(f"RRT path: {len(path_simple)} waypoints")

    print("\nStep 2: Optimizing trajectory...")
    optimized_simple, success_simple, info_simple = helpers.optimize_trajectory(
        rrt_path=path_simple,
        start_pose=start_simple,
        goal_position=goal_simple.translation(),
        hoops=[],
        obstacles=[],
        N=20,
        dt=0.1,
        weights={'thrust': 0.1, 'angular': 1.0, 'smoothness': 5.0},
        initialize_from_rrt_robust=initialize_from_rrt_robust,
        dynamics_constraints_robust=dynamics_constraints_robust,
        boundary_constraints_robust=boundary_constraints_robust,
        collision_constraints_optimized=collision_constraints_optimized,
        cost_function_tuned=cost_function_tuned,
        unpack_decision_vars=unpack_decision_vars
    )

    if success_simple:
        print(f"\n✅ SUCCESS! Cost: {info_simple['cost']:.2f}, Iterations: {info_simple['iterations']}")
    else:
        print(f"\n❌ Optimization failed, using RRT path")

    print("\nStep 3: Visualizing...")
    fig = helpers.visualize_rrt_vs_optimized_comparison(
        path_simple, optimized_simple, start_simple, goal_simple,
        title="Demo 3: RRT vs Optimized"
    )
    fig.show()

    return path_simple, optimized_simple, success_simple


def demo_racing_optimization():
    print("\n" + "=" * 60)
    print("Demo 4: Racing Through Hoops")
    print("=" * 60)

    hoops_demo2 = helpers.get_hoops()
    targets_demo2 = helpers.get_targets()
    start_demo2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))

    print("Step 1: Running RRT through hoops...")
    rrt_race_path = drone_racing_rrt(start_demo2, targets_demo2)
    print(f"RRT path: {len(rrt_race_path)} waypoints")

    print("\nStep 2: Optimizing racing trajectory...")
    optimized_race, success_race, info_race = helpers.optimize_racing_path_sequential(
        rrt_path=rrt_race_path,
        start_pose=start_demo2,
        hoops=hoops_demo2,
        obstacles=[],
        N=25,
        dt=0.1,
        weights={'thrust': 0.05, 'angular': 0.5, 'smoothness': 5.0},
        initialize_from_rrt_robust=initialize_from_rrt_robust,
        dynamics_constraints_robust=dynamics_constraints_robust,
        boundary_constraints_robust=boundary_constraints_robust,
        collision_constraints_optimized=collision_constraints_optimized,
        cost_function_tuned=cost_function_tuned,
        unpack_decision_vars=unpack_decision_vars
    )

    if success_race:
        print(f"\n✅ Racing optimization SUCCESS!")
    else:
        print(f"\n⚠ Some segments may have failed")

    print("\nStep 3: Visualizing...")
    fig = helpers.drone_racing_path_comparison(
        hoops_demo2, start_demo2, rrt_race_path, optimized_race,
        title="Demo 4: RRT vs Optimized Racing"
    )
    fig.show()

    if success_race and 'states' in info_race:
        analyze_trajectory(info_race)

    print("\nPATH COMPARISON:")
    length_rrt = compute_path_length(rrt_race_path)
    length_opt = compute_path_length(optimized_race)
    print(f"  RRT path length: {length_rrt:.2f} m")
    print(f"  Optimized path length: {length_opt:.2f} m")
    print(f"  Reduction: {(length_rrt - length_opt) / length_rrt * 100:.1f}%")
    print(f"\n  RRT waypoints: {len(rrt_race_path)}")
    print(f"  Optimized waypoints: {len(optimized_race)}")

    return rrt_race_path, optimized_race, success_race


def analyze_trajectory(info):
    states_opt = info['states']
    controls_opt = info['controls']

    dt = 0.1
    velocities = np.diff(states_opt[:, :3], axis=0) / dt
    accelerations = np.diff(velocities, axis=0) / dt

    speeds = np.linalg.norm(velocities, axis=1)
    print(f"\nVelocity stats:")
    print(f"   Max speed: {speeds.max():.2f} m/s")
    print(f"   Mean speed: {speeds.mean():.2f} m/s")

    accel_mags = np.linalg.norm(accelerations, axis=1)
    print(f"\nAcceleration stats:")
    print(f"   Max: {accel_mags.max():.2f} m/s²")
    print(f"   Mean: {accel_mags.mean():.2f} m/s²")

    print(f"\nControl stats:")
    print(f"   Thrust range: [{controls_opt[:, 3].min():.1f}, {controls_opt[:, 3].max():.1f}]")
    print(f"   Mean thrust: {controls_opt[:, 3].mean():.1f} (hover=10.0)")

    fig = helpers.plot_velocity_acceleration_profiles(velocities, accelerations, controls_opt, dt)
    fig.show()


def demo_racing_with_obstacles():
    print("\n" + "=" * 80)
    print("🏁 FINAL CHALLENGE: RACING WITH OBSTACLES (EASY)")
    print("=" * 80)

    hoops_final = helpers.get_hoops()
    targets_final = helpers.get_targets()
    start_final = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))
    obstacles_easy = helpers.get_obstacles_easy()

    print(f"\nConfiguration:")
    print(f"   - Start: {start_final.translation()}")
    print(f"   - Hoops: {len(hoops_final)}")
    print(f"   - Obstacles: {len(obstacles_easy)} (EASY)")

    print("\n" + "-" * 80)
    print("STAGE 1: RRT WITH OBSTACLES")
    print("-" * 80)

    start_time = time.time()
    rrt_final_path = drone_racing_rrt_with_obstacles(start_final, targets_final, obstacles_easy)
    rrt_time = time.time() - start_time

    print(f"RRT completed in {rrt_time:.2f}s")
    print(f"Path: {len(rrt_final_path)} waypoints")

    has_collision, _ = helpers.check_path_collision(rrt_final_path, obstacles_easy)
    if not has_collision:
        print("✅ RRT is Collision-free!")
    else:
        print("⚠ RRT may have collisions")

    helpers.drone_racing_path_with_obstacles(
        hoops=hoops_final,
        start=start_final,
        path=rrt_final_path,
        obstacles=obstacles_easy
    )

    print("\n" + "-" * 80)
    print("STAGE 2: TRAJECTORY OPTIMIZATION")
    print("-" * 80)

    opt_start = time.time()
    optimized_final, success_final, info_final = helpers.optimize_racing_path_sequential(
        rrt_path=rrt_final_path,
        start_pose=start_final,
        hoops=hoops_final,
        obstacles=obstacles_easy,
        N=25,
        dt=0.1,
        weights={'thrust': 0.05, 'angular': 0.5, 'smoothness': 5.0},
        initialize_from_rrt_robust=initialize_from_rrt_robust,
        dynamics_constraints_robust=dynamics_constraints_robust,
        boundary_constraints_robust=boundary_constraints_robust,
        collision_constraints_optimized=collision_constraints_optimized,
        cost_function_tuned=cost_function_tuned,
        unpack_decision_vars=unpack_decision_vars
    )
    opt_time = time.time() - opt_start

    if success_final:
        print(f"\n✅ OPTIMIZATION SUCCESS!")
        print(f"   Time: {opt_time:.2f}s")
        print(f"   Total time: {rrt_time + opt_time:.2f}s")

        has_collision_opt, _ = helpers.check_path_collision(optimized_final, obstacles_easy)
        if not has_collision_opt:
            print("✅ Optimized path is collision-free!")
        print(f"✅ Passes through all {len(hoops_final)} hoops")
    else:
        print("\n⚠ Optimization had issues, using RRT path")
        optimized_final = rrt_final_path

    print("\n" + "-" * 80)
    print("STAGE 3: VISUALIZATION")
    print("-" * 80)

    helpers.drone_racing_path_with_obstacles(
        hoops_final, start_final, optimized_final, obstacles_easy
    )

    print("\n✨ Interactive 3D visualization above!")
    print("   - Rotate: mouse drag")
    print("   - Zoom: scroll wheel")
    print("   - Pan: right-click drag")

    length_rrt_final = compute_path_length(rrt_final_path)
    length_opt_final = compute_path_length(optimized_final)

    print("\nPATH COMPARISON:")
    print(f"  RRT path length: {length_rrt_final:.2f} m")
    print(f"  Optimized path length: {length_opt_final:.2f} m")
    print(f"  Reduction: {(length_rrt_final - length_opt_final) / length_rrt_final * 100:.1f}%")
    print(f"\n  RRT waypoints: {len(rrt_final_path)}")
    print(f"  Optimized waypoints: {len(optimized_final)}")

    if success_final:
        print(f"\n✨ The optimized path is smoother and more efficient!")

    return rrt_final_path, optimized_final, success_final


def demo_hard_obstacles_preview():
    print("\n" + "=" * 60)
    print("Demo 6: Hard Obstacles Preview")
    print("=" * 60)

    start_race = gtsam.Pose3(
        r=gtsam.Rot3.Yaw(math.radians(45)),
        t=gtsam.Point3(1, 3, 8)
    )

    print("Visualizing racing environment with 8 obstacles (HARD mode)...")
    helpers.drone_racing_path_with_obstacles(
        helpers.get_hoops(),
        start_race,
        [],
        obstacles=helpers.get_obstacles_hard()
    )


def main():
    print("=" * 80)
    print("🚁 DRONE RACING PATH PLANNING - COMPLETE DEMO")
    print("=" * 80)

    demo_rrt_3d()
    demo_rrt_drone_dynamics()
    demo_simple_optimization()
    demo_racing_optimization()
    demo_racing_with_obstacles()
    demo_hard_obstacles_preview()

    print("\n" + "=" * 80)
    print("🏁 ALL DEMOS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
import math
from typing import List, Tuple, Callable, Union
import numpy as np
import gtsam

rng = np.random.default_rng(12345)


def generate_random_point(target: gtsam.Point3) -> gtsam.Point3:
    if rng.random() < 0.2:
        return target
    return gtsam.Point3(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10))


def distance_euclidean(point1: gtsam.Point3, point2: gtsam.Point3) -> float:
    return np.linalg.norm(point1 - point2)


def find_nearest_node(rrt: List[gtsam.Point3], node: gtsam.Point3) -> Tuple[gtsam.Point3, int]:
    rrt_array = np.array([np.array(p) for p in rrt])
    node_array = np.array(node)
    distances = np.linalg.norm(rrt_array - node_array, axis=1)
    index = np.argmin(distances)
    return rrt[index], index


def steer_naive(parent: gtsam.Point3, target: gtsam.Point3, fraction: float = 0.2) -> gtsam.Point3:
    return parent + fraction * (target - parent)


def run_rrt(start, target, generate_random_node: Callable, steer: Callable, 
            distance: Callable, find_nearest_node: Callable, threshold: float,
            max_iterations: int = 2000) -> Tuple[List, List[int]]:
    rrt = [start]
    parents = [-1]

    for _ in range(max_iterations):
        rand_node = generate_random_node(target)
        nearest_node, index = find_nearest_node(rrt, rand_node)
        steer_node = steer(nearest_node, rand_node)
        rrt.append(steer_node)
        parents.append(index)

        if distance(steer_node, target) < threshold:
            break

    return rrt, parents


def get_rrt_path(rrt: List, parents: List[int]) -> List:
    path = []
    i = len(rrt) - 1
    path.append(rrt[i])

    while parents[i] != -1:
        next_node = rrt[parents[i]]
        path.append(next_node)
        i = parents[i]

    path.reverse()
    return path


def drone_racing_rrt(start: gtsam.Pose3, targets: List[gtsam.Pose3]) -> List[gtsam.Pose3]:
    from .drone_dynamics import generate_random_pose, steer, find_nearest_pose
    from . import helpers_obstacles as helpers

    drone_path = []
    current_start = start

    for i, target in enumerate(targets):
        rrt_tree, parents = run_rrt(
            current_start, target, generate_random_pose, steer,
            helpers.distance_between_poses, find_nearest_pose, threshold=2.0
        )
        path = get_rrt_path(rrt_tree, parents)
        helpers.pass_through_the_hoop(target, path)

        if i == 0:
            drone_path.extend(path)
        else:
            drone_path.extend(path[1:])

        current_start = path[-1]

    return drone_path


def run_rrt_with_obstacles(start, target, generate_random_node: Callable, steer: Callable,
                           distance: Callable, find_nearest_node: Callable, threshold: float,
                           obstacles: List = None, max_iterations: int = 5000) -> Tuple[List, List[int]]:
    from . import helpers_obstacles as helpers

    rrt = [start]
    parents = [-1]
    obstacles = obstacles or []

    for _ in range(max_iterations):
        rand_node = generate_random_node(target)
        nearest_node, n_index = find_nearest_node(rrt, rand_node)
        new_node = steer(nearest_node, rand_node)

        # Extract positions for collision checking
        if isinstance(new_node, gtsam.Pose3):
            point_near = nearest_node.translation()
            point_new = new_node.translation()
        else:
            point_near = nearest_node
            point_new = new_node

        if helpers.check_segment_collision(point_near, point_new, obstacles):
            continue

        rrt.append(new_node)
        parents.append(n_index)

        if distance(new_node, target) < threshold:
            break

    return rrt, parents


def drone_racing_rrt_with_obstacles(start: gtsam.Pose3, targets: List[gtsam.Pose3],
                                    obstacles: List = None) -> List[gtsam.Pose3]:
    from .drone_dynamics import generate_random_pose, steer, find_nearest_pose
    from . import helpers_obstacles as helpers

    drone_path = []
    current_start = start

    for i, target in enumerate(targets):
        rrt_path, parents = run_rrt_with_obstacles(
            current_start, target, generate_random_pose, steer,
            helpers.distance_between_poses, find_nearest_pose,
            threshold=2.0, obstacles=obstacles
        )
        segment_path = get_rrt_path(rrt_path, parents)
        helpers.pass_through_the_hoop(target, segment_path)

        if i == 0:
            drone_path.extend(segment_path)
        else:
            drone_path.extend(segment_path[1:])

        current_start = segment_path[-1]

    return drone_path
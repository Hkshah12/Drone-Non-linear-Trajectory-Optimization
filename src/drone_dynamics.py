import math
from typing import List, Tuple
import numpy as np
import gtsam

rng = np.random.default_rng(12345)


def compute_attitude_from_ypr(yaw: float, pitch: float, roll: float) -> gtsam.Rot3:
    return gtsam.Rot3.RzRyRx(roll, pitch, yaw)


def compute_force(attitude: gtsam.Rot3, thrust: float) -> gtsam.Point3:
    thrust_body = gtsam.Point3(0, 0, thrust)
    return attitude.rotate(thrust_body)


def compute_terminal_velocity(force: gtsam.Point3, drag_coeff: float = 0.5, mass: float = 1.0) -> gtsam.Point3:
    scale = 1.0 / (drag_coeff * mass)
    return gtsam.Point3(force.x() * scale, force.y() * scale, force.z() * scale)


def generate_random_pose(target: gtsam.Pose3) -> gtsam.Pose3:
    if rng.random() < 0.2:
        return target
    
    translation = gtsam.Point3(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10))
    yaw = rng.uniform(math.radians(-60), math.radians(60))
    pitch = rng.uniform(math.radians(-60), math.radians(60))
    roll = rng.uniform(math.radians(-60), math.radians(60))
    rotation = compute_attitude_from_ypr(yaw, pitch, roll)
    
    return gtsam.Pose3(rotation, translation)


def find_nearest_pose(rrt: List[gtsam.Pose3], node: gtsam.Pose3) -> Tuple[gtsam.Pose3, int]:
    rrt_translations = np.array([np.array(p.translation()) for p in rrt])
    node_translation = np.array(node.translation())
    distances = np.linalg.norm(rrt_translations - node_translation, axis=1)
    index = np.argmin(distances)
    return rrt[index], index


def steer_with_terminal_velocity(current: gtsam.Pose3, target: gtsam.Pose3, duration: float = 0.1) -> gtsam.Pose3:
    current_pos = current.translation()
    target_pos = target.translation()
    current_rot = current.rotation()
    
    direction = np.array(target_pos) - np.array(current_pos)
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:
        return current
    
    direction = direction / distance
    desired_velocity = direction * min(distance / duration, 10.0)
    new_pos = current_pos + gtsam.Point3(*desired_velocity) * duration
    new_rot = current_rot.slerp(0.2, target.rotation())
    
    return gtsam.Pose3(new_rot, new_pos)


def compute_force_with_gravity(attitude: gtsam.Rot3, thrust: float, mass: float = 1.0) -> gtsam.Point3:
    g = 10.0
    force = compute_force(attitude, thrust)
    gravity_force = gtsam.Point3(0, 0, -mass * g)
    return force + gravity_force


def steer(current: gtsam.Pose3, target: gtsam.Pose3, duration: float = 0.1) -> gtsam.Pose3:
    yaw_values = [-10, 0, 10]
    pitch_values = [-10, 0, 10]
    roll_values = [-10, 0, 10]
    thrust_values = [5, 10, 15, 20]
    
    current_pos = current.translation()
    target_pos = target.translation()
    current_rot = current.rotation()
    
    best_distance = float('inf')
    best_pose = None
    
    for yaw in yaw_values:
        for pitch in pitch_values:
            for roll in roll_values:
                for thrust in thrust_values:
                    delta_attitude = compute_attitude_from_ypr(
                        math.radians(yaw), math.radians(pitch), math.radians(roll)
                    )
                    new_attitude = current_rot.compose(delta_attitude)
                    force = compute_force_with_gravity(new_attitude, thrust)
                    velocity = compute_terminal_velocity(force)
                    new_pos = current_pos + velocity * duration
                    distance = np.linalg.norm(np.array(target_pos) - np.array(new_pos))
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_pose = gtsam.Pose3(new_attitude, new_pos)
    
    return best_pose
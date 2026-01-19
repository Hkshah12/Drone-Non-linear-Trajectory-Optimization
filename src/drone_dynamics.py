import gtsam
import numpy as np
import math
from typing import List, Tuple
from . import helpers_obstacles as helpers

rng = np.random.default_rng(12345)

def compute_attitude_from_ypr(yaw: float, pitch: float, roll: float) -> gtsam.Rot3:
  '''
  Uses yaw, pitch and roll angles to compute the attitude of the drone
  '''
  return gtsam.Rot3.RzRyRx(roll, pitch, yaw)

def compute_force(attitude: gtsam.Rot3, thrust: float) -> gtsam.Point3:
  '''
  Computes the force vector given attitude and thrust in the body frame
  '''
  thrust_drone = gtsam.Point3(0, 0, thrust)
  force = attitude.rotate(thrust_drone)
  return force

def compute_terminal_velocity(force: gtsam.Point3, kd: float = 0.0425) -> gtsam.Point3:
  '''
  Uses the force vector and drag coefficient to compute the terminal velocity of the drone
  '''
  eps = 1e-6
  force_np = np.array(force)
  force_ed = np.abs(force)
  force_ed += eps

  mag = np.sqrt(force_ed/kd)
  force_np = np.sign(force_np)*mag

  terminal_velocity = gtsam.Point3(force_np)
  return terminal_velocity

def compute_force_with_gravity(attitude: gtsam.Rot3, thrust: float, mass: float = 1.0) -> gtsam.Point3:
  '''
  Computes the net force vector given attitude and thrust in the body frame
  by adjusting for the downwards weight force.
  '''
  g = 10.0  # m/s^2
  force = compute_force(attitude, thrust)
  force_on_drone = gtsam.Point3(0, 0, -mass*g)
  force = force + force_on_drone
  return force

def generate_random_pose(target: gtsam.Pose3) -> gtsam.Pose3:
  '''
  This function generates a random node in the pose configuration space (10x10x10) and returns it.
  '''
  node = None
  
  if rng.random() < 0.2:
    node = target
  else:
    target_pt = gtsam.Point3(rng.uniform(0,10), rng.uniform(0, 10), rng.uniform(0, 10))
    roll = rng.uniform(math.radians(-60), math.radians(60))
    pitch = rng.uniform(math.radians(-60), math.radians(60))
    yaw = rng.uniform(math.radians(-60), math.radians(60))
    
    attitude = compute_attitude_from_ypr(yaw, pitch, roll)
    node = gtsam.Pose3(attitude, target_pt)

  return node

def find_nearest_pose(rrt: List[gtsam.Pose3], node: gtsam.Pose3):
  '''
  This function finds the nearest node in the current RRT tree to the newly sampled node.
  '''
  nearest = None
  index = None
  distance = np.inf

  for i in range(len(rrt)):
    curr_pose1 = rrt[i]
    dist = helpers.distance_between_poses(curr_pose1, node)

    if dist < distance:
      distance = dist
      nearest = curr_pose1
      index = i

  return nearest, index

def steer_with_terminal_velocity(current: gtsam.Pose3, target: gtsam.Pose3, duration: float = 0.1) -> gtsam.Pose3:
  '''
  We need to find a short steering from the current pose toward the target pose.
  '''
  steer_node = None

  current_pos = current.translation()
  target_pos = target.translation()

  direction = target_pos - current_pos
  dir_array = np.array(direction)
  dir_norm = np.linalg.norm(dir_array)
  
  if dir_norm == 0:
      return current

  dir_unit = (dir_array) / (dir_norm)

  attitude = helpers.get_new_attitude(current, dir_unit)
  force = compute_force(attitude, 20.0)
  terminal_velocity = compute_terminal_velocity(force)
  
  new_pos = current_pos + terminal_velocity * duration
  steer_node = gtsam.Pose3(attitude, new_pos)

  return steer_node

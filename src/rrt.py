import numpy as np
import gtsam
import math
from typing import List, Tuple

rng = np.random.default_rng(12345)

def generate_random_point(target: gtsam.Point3) -> gtsam.Point3:
  '''
  This function generates a random node in the 3 dimensional configuration space of (10x10x10) and returns it.
  You must ensure that there is atleast a 20% chance with which the target node itself is returned.
  '''
  node = None
  
  if rng.random() < 0.2:
    node = target
  else:
    node = gtsam.Point3(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10))

  return node

def distance_euclidean(point1: gtsam.Point3, point2: gtsam.Point3) -> float:
  '''
  This function computes the euclidean distance between two 3-D points.
  '''
  return np.linalg.norm(point1 - point2)

def find_nearest_node(rrt: List[gtsam.Point3], node: gtsam.Point3):
  '''
  Given the current RRT tree and the newly sampled node, this function returns the node in the tree which is CLOSEST
  to the newly sampled node, as well as the index of that node.
  '''
  nearest_node = None
  index = None

  rrt_array = np.array([np.array(p) for p in rrt])
  node_array = np.array(node)

  distances = np.linalg.norm(rrt_array - node_array, axis=1)
  index = np.argmin(distances)
  nearest_node = rrt[index]

  return nearest_node, index

def steer_naive(parent: gtsam.Point3, target: gtsam.Point3, fraction = 0.2):
  '''
  Thus function steers the drone towards the target point, going a fraction of the displacement
  It returns the new 'steer_node' which takes us closer to the destination.
  '''
  steer_node = parent + fraction * (target - parent)
  return steer_node

# TODO 5
def run_rrt(start, target, generate_random_node, steer, distance, find_nearest_node, threshold):
  '''
  This function is the main RRT loop and executes the entire RRT algorithm.
  Follow the steps outlined above. You should keep sampling nodes until the terminating condition is met.

  Please use the same function names as given in the function definition.

  Arguments:
   - start: the start node, it could be gtsam.Point3 or gtsam.Pose3.
   - target: the destination node, it could be gtsam.Point3 or gtsam.Pose3.
   - generate_random_node: this function helps us randomly sample a node
   - steer: this function finds the steer node, which takes us closer to our destination
   - distance: this function computes the distance between the two nodes in the tree
   - find_nearest_node: this function finds the nearest node to the randomly sampled node in the tree
   - threshold: float, this is used for the terminating the algorithm

  Returns:
   - rrt: List[gtsam.Point3] or List[gtsam.Pose3], contains the entire tree
   - parents: List[int], contains the index of the parent for each node in the tree
  '''

  rrt = []
  parents = []
  max_iterations = 2000
  rrt.append(start)
  parents.append(-1)

  for i in range(max_iterations):
    ######## Student code here ########

    # Generating a random node in C-space
    rand_node = generate_random_node(target)

    # Next, we need to find the nearest node to the random node in existing tree
    nearest_node, index = find_nearest_node(rrt, rand_node)

    # Steering from nearest node towards the direction of the random node,
    # and then adding new node to existing tree
    steer_node = steer(nearest_node, rand_node)
    rrt.append(steer_node)

    # Index is the parent of the node
    parents.append(index)

    # Distance between steer node (new node) and goal should be lesser than threshold to terminate
    if distance(steer_node, target) < threshold:
      break

  return rrt, parents

def get_rrt_path(rrt: List[gtsam.Pose3], parents: List[int]) -> List[gtsam.Pose3]:
  path = []
  i = len(rrt) - 1
  path.append(rrt[i])

  while(parents[i] != -1):
    next = rrt[parents[i]]
    path.append(next)
    i = parents[i]

  path.reverse()
  return path

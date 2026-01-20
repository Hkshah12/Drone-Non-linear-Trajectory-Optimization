"""Unit tests for all tasks."""

import unittest
import math
import numpy as np
import gtsam

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
    compute_attitude_from_ypr,
    compute_force,
    compute_terminal_velocity,
    generate_random_pose,
    find_nearest_pose,
    steer_with_terminal_velocity,
    compute_force_with_gravity,
    steer
)
from src.trajectory_optimization import (
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
from src import helpers_obstacles as helpers

class TestRRT(unittest.TestCase):
    """Tests for RRT functions."""

    # TODO 1 Test
    def test_generate_random_point(self):
        """Test random point generation stays within bounds."""
        for _ in range(5):
            node = generate_random_point(gtsam.Point3(4, 5, 6))
            self.assertTrue(0 <= node[0] <= 10)
            self.assertTrue(0 <= node[1] <= 10)
            self.assertTrue(0 <= node[2] <= 10)

    # TODO 2 Test
    def test_distance_euclidean(self):
        """Test Euclidean distance calculation."""
        pt1 = gtsam.Point3(2.70109492, 4.55796488, 2.93292049)
        pt2 = gtsam.Point3(4, 7, 2)
        self.assertAlmostEqual(distance_euclidean(pt1, pt2), 2.9190804346571446, places=2)

    # TODO 3 Test
    def test_find_nearest_node(self):
        """Test finding nearest node in RRT tree."""
        pt1 = gtsam.Point3(1, 2, 3)
        pt2 = gtsam.Point3(0.90320894, 3.55218386, 3.71979848)
        pt3 = gtsam.Point3(1.52256715, 4.24174709, 3.37583879)
        pt4 = gtsam.Point3(1.56803165, 4.10257537, 2.795647)
        pt5 = gtsam.Point3(2.68087164, 3.63713802, 4.25464017)
        new_point = gtsam.Point3(3.74935314, 3.2575652, 5.20840562)

        rrt = [pt1, pt2, pt3, pt4, pt5]
        answer, index = find_nearest_node(rrt, new_point)

        self.assertTrue((answer == pt5).all())
        self.assertEqual(index, 4)

    # TODO 4 Test
    def test_steer_naive(self):
        """Test naive steering function."""
        pt1 = gtsam.Point3(3.80319106, 2.49123788, 2.60348781)
        pt2 = gtsam.Point3(3.81712339, 0.33173367, 0.51835128)
        expected = gtsam.Point3(3.80597753, 2.05933704, 2.1864605)

        steer_node = steer_naive(pt1, pt2)

        self.assertTrue(np.allclose(expected, steer_node, atol=1e-2))

    # TODO 5 Test
    def test_run_rrt(self):
        """Test that RRT finds a path to target."""
        start = gtsam.Point3(1, 2, 3)
        target = gtsam.Point3(4, 7, 2)

        rrt, parents = run_rrt(
            start,
            target,
            generate_random_point,
            steer_naive,
            distance_euclidean,
            find_nearest_node,
            threshold=0.5
        )

        # Tree should have nodes
        self.assertGreater(len(rrt), 1)
        # Parents list should match rrt length
        self.assertEqual(len(rrt), len(parents))
        # First parent should be -1 (root)
        self.assertEqual(parents[0], -1)

class TestDroneDynamics(unittest.TestCase):
    """Tests for drone dynamics functions."""

    # TODO 6 Test
    def test_compute_attitude_from_ypr(self):
        """Test attitude computation from yaw, pitch, roll."""
        yaw = math.radians(45)
        pitch = math.radians(30)
        roll = math.radians(60)

        expected_attitude = gtsam.Rot3(
            [0.612372, 0.612372, -0.5],
            [-0.0473672, 0.65974, 0.75],
            [0.789149, -0.435596, 0.433013]
        )
        actual_attitude = compute_attitude_from_ypr(yaw, pitch, roll)

        self.assertTrue(actual_attitude.equals(expected_attitude, tol=1e-2))

    # TODO 7 Test
    def test_compute_force(self):
        """Test force computation from attitude and thrust."""
        attitude = gtsam.Rot3(
            [0.612372, 0.612372, -0.5],
            [-0.0473672, 0.65974, 0.75],
            [0.789149, -0.435596, 0.433013]
        )
        thrust = 20.0

        expected_force = gtsam.Point3(15.78, -8.71, 8.66)
        actual_force = compute_force(attitude, thrust)

        self.assertTrue(np.allclose(actual_force, expected_force, atol=1e-2))

    # TODO 8 Test
    def test_compute_terminal_velocity(self):
        """Test terminal velocity computation."""
        force = gtsam.Point3(15.78, -8.71, 8.66)

        expected_terminal_velocity = gtsam.Point3(19.27, -14.32, 14.27)
        actual_terminal_velocity = compute_terminal_velocity(force)

        self.assertTrue(np.allclose(actual_terminal_velocity, expected_terminal_velocity, atol=1e-2))

class TestSteeringWithTerminalVelocity(unittest.TestCase):
    """Tests for pose-based steering functions."""

    # TODO 9 Test
    def test_generate_random_pose(self):
        """Test random pose generation within bounds."""
        target_node = gtsam.Pose3(
            r=gtsam.Rot3.Yaw(math.radians(45)),
            t=gtsam.Point3(8, 5, 6)
        )

        for _ in range(5):
            random_node = generate_random_pose(target_node)

            # Check translation bounds (0 to 10)
            self.assertTrue(np.all(np.greater_equal(
                random_node.translation(),
                gtsam.Point3(0, 0, 0)
            )))
            self.assertTrue(np.all(np.less_equal(
                random_node.translation(),
                gtsam.Point3(10, 10, 10)
            )))

            # Check rotation bounds (-60 to 60 degrees)
            self.assertTrue(np.all(np.greater_equal(
                random_node.rotation().ypr(),
                gtsam.Point3(math.radians(-60), math.radians(-60), math.radians(-60))
            )))
            self.assertTrue(np.all(np.less_equal(
                random_node.rotation().ypr(),
                gtsam.Point3(math.radians(60), math.radians(60), math.radians(60))
            )))

    # TODO 10 Test
    def test_find_nearest_pose(self):
        """Test finding nearest pose in RRT tree."""
        rrt_tree = [
            gtsam.Pose3(
                r=gtsam.Rot3([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                t=gtsam.Point3(1, 2, 3)
            ),
            gtsam.Pose3(
                r=gtsam.Rot3([0.771517, -0.617213, 0],
                             [0.0952381, 0.119048, -0.97619],
                             [0.617213, 0.771517, 0.154303]),
                t=gtsam.Point3(2.70427, 3.90543, 3.85213)
            ),
            gtsam.Pose3(
                r=gtsam.Rot3([0.601649, -0.541882, 0.302815],
                             [-0.301782, -0.62385, -0.516772],
                             [0.627501, 0.29376, -0.721074]),
                t=gtsam.Point3(4.42268, 5.08119, 2.01005)
            ),
            gtsam.Pose3(
                r=gtsam.Rot3([-0.696943, 0.589581, -0.36631],
                             [-0.664345, -0.416218, 0.594076],
                             [0.204431, 0.679463, 0.704654]),
                t=gtsam.Point3(5.40351, 6.86933, 3.83104)
            ),
            gtsam.Pose3(
                r=gtsam.Rot3([-0.0686996, 0.218721, -0.818805],
                             [-0.796488, -0.297401, -0.0126152],
                             [-0.340626, 0.900832, 0.269211]),
                t=gtsam.Point3(1.43819, 5.96437, 4.97769)
            )
        ]

        new_node = gtsam.Pose3(
            r=gtsam.Rot3([0.682707, 0.661423, 0.310534],
                         [-0.626039, 0.748636, -0.218217],
                         [-0.376811, -0.0454286, 0.925176]),
            t=gtsam.Point3(5.65333, 5.65964, 1.60624)
        )

        expected_nearest_node = rrt_tree[2]
        expected_index = 2

        actual_nearest_node, actual_index = find_nearest_pose(rrt_tree, new_node)

        self.assertTrue(actual_nearest_node.equals(expected_nearest_node, tol=1e-1))
        self.assertEqual(actual_index, expected_index)

    def test_steer_with_terminal_velocity(self):
        """Test steering with terminal velocity dynamics."""
        current_node = gtsam.Pose3(
            gtsam.Rot3.Yaw(math.radians(90)),
            gtsam.Point3(1, 2, 3)
        )
        new_node = gtsam.Pose3(
            gtsam.Rot3.Pitch(math.radians(45)),
            gtsam.Point3(8, 5, 6)
        )

        expected_steer_node = gtsam.Pose3(
            gtsam.Rot3([0.37, -0.86, 0],
                       [0.31, 0.13, -0.87],
                       [0.86, 0.37, 0.37]),
            gtsam.Point3(3.00, 3.31, 4.31)
        )

        actual_steer_node = steer_with_terminal_velocity(current_node, new_node)

        self.assertTrue(actual_steer_node.equals(expected_steer_node, tol=1e-1))


class TestRealisticSteer(unittest.TestCase):
    """Tests for realistic steering with gravity."""

    # TODO 12 Test
    def test_compute_force_with_gravity(self):
        """Test force computation including gravity."""
        attitude = gtsam.Rot3(
            [0.612372, 0.612372, -0.5],
            [-0.0473672, 0.65974, 0.75],
            [0.789149, -0.435596, 0.433013]
        )
        thrust = 20.0

        expected_force = gtsam.Point3(15.78, -8.71, -1.34)
        actual_force = compute_force_with_gravity(attitude, thrust)

        self.assertTrue(np.allclose(actual_force, expected_force, atol=1e-2))

    # TODO 13 Test
    def test_steer(self):
        """Test realistic steering function."""
        current_node = gtsam.Pose3(
            gtsam.Rot3.Yaw(math.radians(90)),
            gtsam.Point3(1, 2, 3)
        )
        new_node = gtsam.Pose3(
            gtsam.Rot3.Pitch(math.radians(45)),
            gtsam.Point3(8, 5, 6)
        )

        expected_steer_node = gtsam.Pose3(
            gtsam.Rot3([0.17, 0.97, -0.17],
                       [-0.96, 0.20, 0.17],
                       [0.20, 0.14, 0.97]),
            gtsam.Point3(1.97, 2.81, 4.49)
        )
        actual_steer_node = steer(current_node, new_node)

        self.assertTrue(actual_steer_node.equals(expected_steer_node, tol=1e-2))

class TestDroneRacingAndObstacles(unittest.TestCase):
    """Tests for drone racing and obstacle avoidance functions."""

    # TODO 14 Test
    def test_drone_racing_rrt(self):
        """Test drone racing through multiple hoops."""
        start = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
        targets = [
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(3, 4, 5)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(6, 7, 8))
        ]
        
        path = drone_racing_rrt(start, targets)
        
        # Path should exist and have multiple waypoints
        self.assertGreater(len(path), 2)

    # TODO 15 Test
    def test_run_rrt_with_obstacles_no_obstacles(self):
        """Test RRT with obstacles behaves like regular RRT when no obstacles."""
        start = gtsam.Point3(1, 2, 3)
        target = gtsam.Point3(4, 7, 2)
        
        rrt, parents = run_rrt_with_obstacles(
            start,
            target,
            generate_random_point,
            steer_naive,
            distance_euclidean,
            find_nearest_node,
            threshold=0.5,
            obstacles=[]
        )
        
        # Should find a path
        self.assertGreater(len(rrt), 1)
        self.assertEqual(len(rrt), len(parents))

    def test_run_rrt_with_obstacles_avoids_collision(self):
        """Test RRT avoids obstacles."""
        start = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 5))
        target = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 8))
        
        # Create obstacle in the middle
        obstacles = [
            helpers.SphereObstacle(center=[5, 5, 6.5], radius=1.5, name="Central Pillar")
        ]
        
        rrt, parents = run_rrt_with_obstacles(
            start,
            target,
            generate_random_pose,
            steer,
            helpers.distance_between_poses,
            find_nearest_pose,
            threshold=1.5,
            obstacles=obstacles
        )
        
        # Should find a path
        self.assertGreater(len(rrt), 1)
        
        # Path should be collision-free
        path = get_rrt_path(rrt, parents)
        has_collision, _ = helpers.check_path_collision(path, obstacles)
        self.assertFalse(has_collision)

    def test_drone_racing_rrt_with_obstacles(self):
        """Test drone racing through hoops while avoiding obstacles."""
        start = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
        targets = [
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(4, 4, 5)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(7, 7, 7))
        ]
        obstacles = [
            helpers.SphereObstacle(center=[3, 3, 4], radius=0.5, name="Obstacle1")
        ]
        
        path = drone_racing_rrt_with_obstacles(start, targets, obstacles)
        
        # Path should exist and have multiple waypoints
        self.assertGreater(len(path), 2)
        
        # Path should be collision-free
        has_collision, _ = helpers.check_path_collision(path, obstacles)
        self.assertFalse(has_collision)

class TestTrajOptHelpers(unittest.TestCase):
    """Test angle wrapping, packing, and unpacking functions."""

    def test_angle_diff_small(self):
        """Test angle_diff with small differences."""
        # Small positive difference
        result = angle_diff(0.1, 0.0)
        self.assertAlmostEqual(result, 0.1, places=6)

        # Small negative difference
        result = angle_diff(0.0, 0.1)
        self.assertAlmostEqual(result, -0.1, places=6)

    def test_angle_diff_wrapping(self):
        """Test angle_diff with wrapping around ±π."""
        # Wrapping across +π/-π boundary
        result = angle_diff(np.pi - 0.1, -np.pi + 0.1)
        self.assertAlmostEqual(result, 0.2, places=6)

        # Wrapping the other direction
        result = angle_diff(-np.pi + 0.1, np.pi - 0.1)
        self.assertAlmostEqual(result, -0.2, places=6)

    def test_angle_diff_180_degrees(self):
        """Test angle_diff at exactly 180 degrees."""
        # Exactly π apart (ambiguous, but should handle consistently)
        result = angle_diff(0.0, np.pi)
        self.assertTrue(abs(result - np.pi) < 1e-6 or abs(result + np.pi) < 1e-6)

    def test_pack_decision_vars(self):
        """Test packing states and controls into flat vector."""
        N = 2
        states = np.array([[1, 2, 3, 0.1, 0.2, 0.3],
                           [4, 5, 6, 0.4, 0.5, 0.6],
                           [7, 8, 9, 0.7, 0.8, 0.9]])  # (N+1) x 6
        controls = np.array([[0.01, 0.02, 0.03, 10],
                             [0.04, 0.05, 0.06, 12]])  # N x 4

        z = pack_decision_vars(states, controls, N)

        # Check size
        self.assertEqual(z.shape[0], 26)  # 10*2 + 6 = 26

        # Check first state
        self.assertTrue(np.allclose(z[:6], [1, 2, 3, 0.1, 0.2, 0.3]))

        # Check last state
        self.assertTrue(np.allclose(z[12:18], [7, 8, 9, 0.7, 0.8, 0.9]))

        # Check first control
        self.assertTrue(np.allclose(z[18:22], [0.01, 0.02, 0.03, 10]))

        # Check second control
        self.assertTrue(np.allclose(z[22:26], [0.04, 0.05, 0.06, 12]))

    def test_unpack_decision_vars(self):
        """Test unpacking flat vector into states and controls."""
        N = 2
        # Create a known flat vector
        z = np.array([1, 2, 3, 0.1, 0.2, 0.3,   # state 0
                      4, 5, 6, 0.4, 0.5, 0.6,   # state 1
                      7, 8, 9, 0.7, 0.8, 0.9,   # state 2
                      0.01, 0.02, 0.03, 10,     # control 0
                      0.04, 0.05, 0.06, 12])    # control 1

        states, controls = unpack_decision_vars(z, N)

        # Check shapes
        self.assertEqual(states.shape, (3, 6))
        self.assertEqual(controls.shape, (2, 4))

        # Check first state
        self.assertTrue(np.allclose(states[0, :], [1, 2, 3, 0.1, 0.2, 0.3]))

        # Check last state
        self.assertTrue(np.allclose(states[2, :], [7, 8, 9, 0.7, 0.8, 0.9]))

        # Check controls
        self.assertTrue(np.allclose(controls[0, :], [0.01, 0.02, 0.03, 10]))
        self.assertTrue(np.allclose(controls[1, :], [0.04, 0.05, 0.06, 12]))

    def test_pack_unpack_inverse(self):
        """Test that pack and unpack are inverse operations."""
        N = 3
        # Create random states and controls
        np.random.seed(42)
        states = np.random.randn(N + 1, 6)
        controls = np.random.randn(N, 4)

        # Pack then unpack
        z = pack_decision_vars(states, controls, N)
        states_recovered, controls_recovered = unpack_decision_vars(z, N)

        # Should recover original arrays
        self.assertTrue(np.allclose(states, states_recovered))
        self.assertTrue(np.allclose(controls, controls_recovered))

class TestCostFunctions(unittest.TestCase):
    """Test all cost functions for trajectory optimization."""

    # TODO 20 Test
    def test_cost_function_thrust_hover(self):
        """Test thrust cost at hover condition."""
        N = 2
        states = np.zeros((N + 1, 6))
        # All thrust at hover (10 N)
        controls = np.array([[0, 0, 0, 10],
                             [0, 0, 0, 10]])

        cost = cost_function_thrust(states, controls, N, weight=0.1)
        # Thrust exactly at hover, so cost should be 0
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_cost_function_thrust_deviation(self):
        """Test thrust cost with deviation from hover."""
        N = 2
        states = np.zeros((N + 1, 6))
        # Thrust deviates by +2 from hover (10 -> 12)
        controls = np.array([[0, 0, 0, 12],
                             [0, 0, 0, 12]])

        cost = cost_function_thrust(states, controls, N, weight=0.1)
        # Cost = 0.1 * (2^2 + 2^2) = 0.1 * 8 = 0.8
        self.assertAlmostEqual(cost, 0.8, places=6)

    # TODO 21 Test
    def test_cost_function_angular_zero(self):
        """Test angular cost with zero angular velocities."""
        N = 2
        states = np.zeros((N + 1, 6))
        controls = np.array([[0, 0, 0, 10],
                             [0, 0, 0, 10]])

        cost = cost_function_angular(states, controls, N, weight=1.0)
        # All angular velocities zero, cost should be 0
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_cost_function_angular_nonzero(self):
        """Test angular cost with non-zero angular velocities."""
        N = 2
        states = np.zeros((N + 1, 6))
        # Angular velocities: [0.1, 0.2, 0.3]
        controls = np.array([[0.1, 0.2, 0.3, 10],
                             [0.1, 0.2, 0.3, 10]])

        cost = cost_function_angular(states, controls, N, weight=1.0)
        # Cost = 1.0 * ((0.1^2 + 0.2^2 + 0.3^2) + (0.1^2 + 0.2^2 + 0.3^2))
        #      = 1.0 * (0.14 + 0.14) = 0.28
        self.assertAlmostEqual(cost, 0.28, places=6)

    # TODO 22 Test
    def test_cost_function_smoothness_constant(self):
        """Test smoothness cost with constant controls."""
        N = 3
        states = np.zeros((N + 1, 6))
        # Constant controls (no jerk)
        controls = np.array([[0.1, 0.1, 0.1, 10],
                             [0.1, 0.1, 0.1, 10],
                             [0.1, 0.1, 0.1, 10]])

        cost = cost_function_smoothness(states, controls, N, weight=5.0)
        # Controls don't change, so smoothness cost should be 0
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_cost_function_smoothness_varying(self):
        """Test smoothness cost with varying controls."""
        N = 2
        states = np.zeros((N + 1, 6))
        # Controls change from k=0 to k=1
        controls = np.array([[0, 0, 0, 10],
                             [0.1, 0.1, 0.1, 12]])

        cost = cost_function_smoothness(states, controls, N, weight=5.0)
        # Difference: [0.1, 0.1, 0.1, 2]
        # Cost = 5.0 * (0.1^2 + 0.1^2 + 0.1^2 + 2^2) = 5.0 * 4.03 = 20.15
        self.assertAlmostEqual(cost, 20.15, places=6)

    def test_cost_function_gimbal_lock_safe(self):
        """Test gimbal lock penalty in safe range."""
        N = 2
        # Pitch within safe range (< 50°)
        states = np.array([[0, 0, 0, 0, 0.5, 0],      # pitch = 0.5 rad (~29°) - OK
                           [0, 0, 0, 0, 0.6, 0],      # pitch = 0.6 rad (~34°) - OK
                           [0, 0, 0, 0, 0.7, 0]])     # pitch = 0.7 rad (~40°) - OK
        controls = np.zeros((N, 4))

        cost = cost_function_gimbal_lock(states, controls, N)
        # All pitches within safe range, cost should be 0
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_cost_function_gimbal_lock_danger(self):
        """Test gimbal lock penalty approaching singularity."""
        N = 2
        pitch_limit = 50 * np.pi / 180  # ~0.873 rad
        # Pitch exceeds safe range
        states = np.array([[0, 0, 0, 0, 1.0, 0],      # pitch = 1.0 rad (~57°) - DANGER
                           [0, 0, 0, 0, 0.5, 0],      # pitch = 0.5 rad - OK
                           [0, 0, 0, 0, -1.0, 0]])    # pitch = -1.0 rad - DANGER
        controls = np.zeros((N, 4))

        cost = cost_function_gimbal_lock(states, controls, N)
        self.assertTrue(cost > 30.0)  # Should have significant penalty

    def test_cost_function_integration(self):
        """Test integrated cost function combines all costs."""
        N = 2
        states = np.array([[0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 0.1, 0.1, 0.1],
                           [2, 2, 2, 0.2, 0.2, 0.2]])
        controls = np.array([[0.1, 0.1, 0.1, 12],
                             [0.1, 0.1, 0.1, 12]])

        z = pack_decision_vars(states, controls, N)
        weights = {'thrust': 0.1, 'angular': 1.0, 'smoothness': 5.0}

        cost = cost_function_tuned(z, N, weights)

        # Should be sum of individual costs
        cost_thrust = cost_function_thrust(states, controls, N, weights['thrust'])
        cost_angular = cost_function_angular(states, controls, N, weights['angular'])
        cost_smooth = cost_function_smoothness(states, controls, N, weights['smoothness'])
        cost_gimbal = cost_function_gimbal_lock(states, controls, N)

        expected_cost = cost_thrust + cost_angular + cost_smooth + cost_gimbal
        self.assertAlmostEqual(cost, expected_cost, places=6)

class TestConstraints(unittest.TestCase):
    """Test dynamics and boundary constraint functions."""

    # TODO 25 Tests
    def test_dynamics_constraints_hover(self):
        """Test dynamics constraints for hovering (stationary) trajectory."""
        N = 2
        dt = 0.1

        # Hovering: same position, zero attitude, hover thrust
        states = np.array([[5, 5, 5, 0, 0, 0],
                           [5, 5, 5, 0, 0, 0],
                           [5, 5, 5, 0, 0, 0]])
        controls = np.array([[0, 0, 0, 10],    # Zero angular changes, hover thrust
                             [0, 0, 0, 10]])

        z = pack_decision_vars(states, controls, N)
        violations = dynamics_constraints_robust(z, N, dt)

        # Check shape: should be 6*N = 12 constraints
        self.assertEqual(violations.shape[0], 6 * N)

        # Hovering should nearly satisfy dynamics (small violations due to drag)
        self.assertTrue(np.max(np.abs(violations)) < 1.0)

    def test_dynamics_constraints_forward_flight(self):
        """Test dynamics constraints for forward flight."""
        N = 2
        dt = 0.1

        # Forward flight: moving in +x direction with pitch
        states = np.array([[0, 0, 5, 0, 0.2, 0],  # pitch forward 0.2 rad
                           [1, 0, 5, 0, 0.2, 0],  # moved 1m forward
                           [2, 0, 5, 0, 0.2, 0]]) # moved 2m total
        controls = np.array([[0, 0, 0, 15],       # More thrust for forward flight
                             [0, 0, 0, 15]])

        z = pack_decision_vars(states, controls, N)
        violations = dynamics_constraints_robust(z, N, dt)

        # Check shape
        self.assertEqual(violations.shape[0], 6 * N)

        # Violations won't be zero (we didn't compute exact dynamics)
        # But they should exist (we're testing the function runs)
        self.assertTrue(isinstance(violations, np.ndarray))

    def test_boundary_constraints_start(self):
        """Test boundary constraints enforce start pose."""
        N = 2
        start_pose = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.2, 0.3),
                                 gtsam.Point3(1, 2, 3))
        goal_position = np.array([8, 9, 10])

        # States that match start pose
        states = np.array([[1, 2, 3, 0.1, 0.2, 0.3],  # Matches start
                           [4, 5, 6, 0.1, 0.2, 0.3],
                           [8, 9, 10, 0.1, 0.2, 0.3]])  # Matches goal position
        controls = np.zeros((N, 4))

        z = pack_decision_vars(states, controls, N)
        violations = boundary_constraints_robust(z, N, start_pose, goal_position, [])

        # Check shape: 6 (start) + 3 (goal) + 0 (hoops) = 9
        self.assertEqual(violations.shape[0], 9)

        # First 6 violations (start pose) should be near zero
        self.assertTrue(np.max(np.abs(violations[:6])) < 0.1)

        # Last 3 violations (goal position) should be near zero
        self.assertTrue(np.max(np.abs(violations[6:9])) < 0.1)

    def test_boundary_constraints_with_hoops(self):
        """Test boundary constraints with hoop waypoints."""
        N = 10
        start_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
        goal_position = np.array([10, 0, 0])

        # Create states with 2 hoops at specific locations
        states = np.zeros((N + 1, 6))
        states[:, 0] = np.linspace(0, 10, N + 1)  # x from 0 to 10
        # Make sure some states pass near hoop positions
        states[3, :] = [3, 5, 5, 0, 0, 0]   # Near hoop 1
        states[7, :] = [7, 8, 8, 0, 0, 0]   # Near hoop 2
        states[N, :3] = goal_position        # Goal position

        controls = np.zeros((N, 4))

        # Define 2 hoops
        hoop1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(3, 5, 5))
        hoop2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(7, 8, 8))
        hoops = [hoop1, hoop2]

        z = pack_decision_vars(states, controls, N)
        violations = boundary_constraints_robust(z, N, start_pose, goal_position, hoops)

        # Check shape: 6 (start) + 3 (goal) + 3*2 (hoops) = 15
        self.assertEqual(violations.shape[0], 15)

        # All violations should exist (function runs correctly)
        self.assertTrue(isinstance(violations, np.ndarray))

    def test_boundary_constraints_dimensions(self):
        """Test boundary constraints have correct dimensions for various hoop counts."""
        N = 5
        start_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0))
        goal_position = np.array([5, 5, 5])

        np.random.seed(42)
        states = np.random.randn(N + 1, 6)
        states[0, :] = [0, 0, 0, 0, 0, 0]
        states[N, :3] = goal_position
        controls = np.zeros((N, 4))
        z = pack_decision_vars(states, controls, N)

        # Test with 0 hoops
        violations = boundary_constraints_robust(z, N, start_pose, goal_position, [])
        self.assertEqual(violations.shape[0], 9)  # 6 + 3 + 0

        # Test with 1 hoop
        hoop1 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 2))
        violations = boundary_constraints_robust(z, N, start_pose, goal_position, [hoop1])
        self.assertEqual(violations.shape[0], 12)  # 6 + 3 + 3

        # Test with 3 hoops
        hoop2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(3, 3, 3))
        hoop3 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(4, 4, 4))
        violations = boundary_constraints_robust(z, N, start_pose, goal_position, [hoop1, hoop2, hoop3])
        self.assertEqual(violations.shape[0], 18)  # 6 + 3 + 9

class TestCollisionAndInitialization(unittest.TestCase):
    """Test collision constraints and RRT initialization."""

    # TODO 27 Test
    def test_collision_constraints_no_obstacles(self):
        """Test collision constraints with no obstacles."""
        N = 5
        states = np.zeros((N + 1, 6))
        states[:, 0] = np.linspace(0, 10, N + 1)
        controls = np.zeros((N, 4))
        controls[:, 3] = 10.0  # Hover thrust
        
        z = pack_decision_vars(states, controls, N)
        violations = collision_constraints_optimized(z, N, obstacles=[])
        
        # No obstacles = no violations
        self.assertEqual(len(violations), 0)

    def test_collision_constraints_with_sphere(self):
        """Test collision constraints detect sphere collision."""
        N = 5
        states = np.zeros((N + 1, 6))
        states[:, 0] = np.linspace(0, 10, N + 1)  # x from 0 to 10
        states[:, 1] = 5  # y = 5
        states[:, 2] = 5  # z = 5
        controls = np.zeros((N, 4))
        controls[:, 3] = 10.0
        
        # Obstacle at (5, 5, 5) - directly in path
        obstacles = [helpers.SphereObstacle(center=[5, 5, 5], radius=1.0, name="Test")]
        
        z = pack_decision_vars(states, controls, N)
        violations = collision_constraints_optimized(z, N, obstacles, subsample=1)
        
        # Should have some positive violations (collision)
        self.assertTrue(np.any(violations > 0))

    # TODO 28 Test
    def test_initialize_from_rrt_basic(self):
        """Test RRT initialization produces valid decision vector."""
        # Create simple RRT path
        rrt_path = [
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 2)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(5, 5, 5)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 8)),
            gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(10, 10, 10))
        ]
        N = 10
        dt = 0.1
        start_pose = rrt_path[0]
        
        z = initialize_from_rrt_robust(rrt_path, N, dt, start_pose)
        
        # Check vector size
        expected_size = 6 * (N + 1) + 4 * N  # states + controls
        self.assertEqual(len(z), expected_size)
        
        # Unpack and verify
        states, controls = unpack_decision_vars(z, N)
        
        # Check shapes
        self.assertEqual(states.shape, (N + 1, 6))
        self.assertEqual(controls.shape, (N, 4))
        
        # Check start position matches
        np.testing.assert_array_almost_equal(states[0, :3], [0, 0, 0], decimal=2)
        
        # Check end position is interpolated correctly
        np.testing.assert_array_almost_equal(states[N, :3], [10, 10, 10], decimal=2)

    def test_initialize_controls_bounded(self):
        """Test that initialized controls are within bounds."""
        rrt_path = [
            gtsam.Pose3(gtsam.Rot3.Ypr(0, 0, 0), gtsam.Point3(0, 0, 0)),
            gtsam.Pose3(gtsam.Rot3.Ypr(0.5, 0.3, 0.2), gtsam.Point3(5, 5, 5)),
            gtsam.Pose3(gtsam.Rot3.Ypr(1.0, 0.5, 0.3), gtsam.Point3(10, 10, 10))
        ]
        N = 10
        dt = 0.1
        start_pose = rrt_path[0]
        
        z = initialize_from_rrt_robust(rrt_path, N, dt, start_pose)
        states, controls = unpack_decision_vars(z, N)
        
        # Check angular controls bounded
        deg_to_rad = np.pi / 180
        self.assertTrue(np.all(controls[:, 0:3] >= -10 * deg_to_rad))
        self.assertTrue(np.all(controls[:, 0:3] <= 10 * deg_to_rad))
        
        # Check thrust bounded
        self.assertTrue(np.all(controls[:, 3] >= 5))
        self.assertTrue(np.all(controls[:, 3] <= 20))

if __name__ == "__main__":
    unittest.main()
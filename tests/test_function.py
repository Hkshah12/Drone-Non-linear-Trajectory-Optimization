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
    get_rrt_path
)
from src.drone_dynamics import (
    compute_attitude_from_ypr,
    compute_force,
    compute_terminal_velocity,
    generate_random_pose,
    find_nearest_pose,
    steer_with_terminal_velocity
)


# =============================================================================
# RRT Tests (TODO 1-5)
# =============================================================================
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


# =============================================================================
# Drone Dynamics Tests (TODO 6-8)
# =============================================================================
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


# =============================================================================
# Steering with Terminal Velocity Tests (TODO 9-11)
# =============================================================================
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

    # TODO 11 Test
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


# =============================================================================
# TODO 12-28 Tests (Add as you implement)
# =============================================================================


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
"""Provide geometric path planning functions."""

import numpy as np

from .utils import interpolate_quat, rotvec_to_quat
from .utils import quat_to_matrix, matrix_to_quat


def get_arc_path(start, axis, path_duration, num_points):
    """Get a circular path between two points.
    Args:
        start: The starting point of the path.
        axis: The axis of the circle. [x, y, z, rx, ry, rz]
              First 3 elements are the position of the axis.
              Next 3 elements are the rotation in rotation vector format.
        path_duration: The duration of the path.
        num_points: The number of points in the path.
    """
    assert path_duration > 0, "Path duration must be greater than 0."
    assert num_points > 2, "Number of points must be greater than 2."
    start = np.array(start)
    axis = np.array(axis)

    # Interpolate the path
    path = []
    for point_idx in range(num_points):
        u = point_idx / (num_points - 1)
        # Get the time of the point
        time_t = path_duration * u
        # Get the points on the line by interpolation
        point = arc_interpolate(start, axis, u)
        path.append((time_t, point))

    return path


def get_linear_path(start, goal, path_duration, num_points):
    """Get a linear path between two points.
    Directly run linear interpolation between two points to get the path.
    Args:
        start: The starting point of the path.
        goal: The goal point of the path.
        path_duration: The duration of the path.
        num_points: The number of points in the path
    """
    assert path_duration > 0, "Path duration must be greater than 0."
    assert num_points > 2, "Number of points must be greater than 2."
    start = np.array(start)
    goal = np.array(goal)

    # Interpolate the path
    path = []
    for point_idx in range(num_points):
        u = point_idx / (num_points - 1)
        # Get the time of the point
        time_t = path_duration * u
        # Get the points on the line by interpolation
        point = linear_interpolate(start, goal, u)
        path.append((time_t, point))

    return path


def arc_interpolate(start, axis, u):
    """Interpolate between two workspace points."""
    # Compute the rotation quaternion for the arc
    rot_pos = axis[:3]
    rot_quat = rotvec_to_quat(axis[3:] * u)

    # Position interpolation
    # rotate the start point by the rotation quaternion
    point = np.dot(quat_to_matrix(rot_quat), start[:3] - rot_pos) + rot_pos

    # Rotation interpolation
    # rotate the start quaternion by the rotation quaternion
    if len(start) > 3:
        quat = matrix_to_quat(
            np.dot((quat_to_matrix(rot_quat)), quat_to_matrix(start[3:7]))
        )
        return np.concatenate([point, quat])

    return point


def linear_interpolate(start, goal, u):
    """Interpolate between two workspace points."""
    assert start.shape == goal.shape

    # Position interpolation
    point = start[:3] + u * (goal[:3] - start[:3])

    # Rotation interpolation
    if len(start) > 3:
        quat = interpolate_quat(start[3:7], goal[3:7], u)
        point = np.concatenate([point, quat])

    return point

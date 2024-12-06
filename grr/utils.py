"""Utility functions for GRR"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sklearn.neighbors import BallTree


def se3_distance(p1, p2, position_weight=1.0, rotation_weight=0.3):
    """Compute the distance between two workspace SE3 points."""
    # Position component
    # position distance - euclidean
    d_position = math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

    # Rotation component
    # # rotation distance - arc length
    # d_rotation = np.abs(np.dot(point1[3:7], point2[3:7]))
    # if d_rotation > 1:
    #     d_rotation = 1  # Clip to [0, 1] range
    # d_rotation = 2 * np.arccos(d_rotation)

    # rotation distance - simplified
    # d_rotation = 1 - np.abs(np.dot(point1[3:7], point2[3:7]))
    d_rotation = 1 - abs(
        p1[3] * p2[3] + p1[4] * p2[4] + p1[5] * p2[5] + p1[6] * p2[6]
    )

    return position_weight * d_position + rotation_weight * d_rotation


def quaternion_angle(q1, q2):
    """Compute the distance between two SO(3) points (quaternion)
    This function is equivalent to the arc length distance in rotation
    """
    # Cos distance
    dist = np.min([np.abs(np.dot(q1, q2)), 1.0])  # prevent numerical error
    angle = 2 * np.arccos(dist)
    return angle


def quaternion_close(q1, q2, eps=1e-3):
    """Check if two quaternions are close to each other"""
    return quaternion_angle(q1, q2) < eps


def interpolate_quat(quat1, quat2, u):
    """Interpolate between two rotation vectors given a ratio"""
    quat1 = R.from_quat(quat1).as_quat()
    quat2 = R.from_quat(quat2).as_quat()

    # Spherical linear interpolation (SLERP)
    rotations = R.from_quat([quat1, quat2])
    slerp = Slerp([0, 1], rotations)
    interpolated_quat = slerp([u])[0].as_quat()

    return interpolated_quat


def quat_to_matrix(quat):
    """Convert a quaternion to a rotation matrix"""
    return R.from_quat(quat).as_matrix()


def euler_to_matrix(euler, seq="xyz", degrees=False):
    """Convert euler angles to a rotation matrix"""
    return R.from_euler(seq, euler, degrees).as_matrix()


def matrix_to_quat(matrix):
    """Convert a rotation matrix to a quaternion"""
    if len(matrix) == 9:
        matrix = np.reshape(matrix, (3, 3))
    return R.from_matrix(matrix).as_quat()


def quat_to_euler(quat, seq="xyz", degrees=False):
    """Convert quaternion to an euler angles"""
    return R.from_quat(quat).as_euler(seq, degrees)


def rotvec_to_quat(rotvec):
    """Convert euler angles to a quaternion"""
    return R.from_rotvec(rotvec).as_quat()


def quat_to_rotvec(quat):
    """Convert a quaternion to a rotation vector"""
    return R.from_quat(quat).as_rotvec()


def euler_to_quat(euler, seq="xyz", degrees=False):
    """Convert euler angles to a quaternion"""
    return R.from_euler(seq, euler, degrees).as_quat()


def wrap_to_pi(angle):
    """Wrap an angle to [-Pi, Pi]"""
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle


def interpolate_angle(angle1, angle2, u):
    """Interpolate between two angles given a ratio"""
    # Wrap delta within [-Pi, Pi] for shortest path interpolation
    delta = wrap_to_pi(angle2 - angle1)
    # Interpolate and wrap the result
    result = wrap_to_pi(angle1 + u * delta)

    return result


def sample_quat():
    """Sample a quaternion with uniform random distribution"""
    return R.random().as_quat()


def get_staggered_grid(n_points, domain):
    """Get n_points workspace points using staggered grid sampling

    Args:
        n_points: number of points to sample
        domain: domain of the workspace
    Returns:
        points: a list of workspace points (coordinates)
        edges: a list of edges in index form
               (i, j), ith point connected to jth point
    """
    # Keep a record of the domain that is constant
    constant_domain = {}
    for i, domain_i in enumerate(domain):
        if domain_i[0] == domain_i[1]:
            constant_domain[i] = domain_i[0]

    # Calculate the range and proportion of each dimension
    dim_ranges = np.array(
        [
            domain_i[1] - domain_i[0]
            for domain_i in domain
            if domain_i[0] != domain_i[1]
        ]
    )
    dim_prop = dim_ranges / np.sum(dim_ranges)

    # Compute how many corner points and center points should be assigned
    # solve n_corners + n_centers = n_points
    # with simple approximation, assuming n_corners ~= n_centers
    n_corners = round(n_points / 2)

    # Based on proportion, compute the # point in each dimension
    # solve p: (a1 * p) * (a2 * p) * ... * (ak * p) = n_corners
    # where ak is the proportion, ak != 0, and
    # ak * p is the number of points in the kth dimension
    p = np.power(n_corners / np.prod(dim_prop), 1 / len(dim_prop))
    corner_points_per_dim = [round(proportion * p) for proportion in dim_prop]

    # Ensure # corner points in each dimension is an odd number
    corner_points_per_dim = [
        p + 1 if p % 2 == 0 else p for p in corner_points_per_dim
    ]

    # Calculate the spacing of corners
    # spacing should be the same for all dimensions (except 0)
    spacing = dim_ranges[0] / (corner_points_per_dim[0] - 1)

    # Calculate a list of points along each dimension for grid corner
    corners = [
        np.linspace(domain[i][0], domain[i][1], corner_points_per_dim[i])
        for i in range(len(dim_prop))
    ]
    # Calculate a list of points along each dimension for grid center
    centers = [
        np.linspace(
            domain[i][0] + spacing / 2,
            domain[i][1] - spacing / 2,
            corner_points_per_dim[i] - 1,
        )
        for i in range(len(dim_prop))
    ]

    # Generate all combinations of points across dimensions
    corner_mesh = np.meshgrid(*corners)
    center_mesh = np.meshgrid(*centers)
    corner_points = np.vstack(list(map(np.ravel, corner_mesh))).T
    center_points = np.vstack(list(map(np.ravel, center_mesh))).T
    # add the constant domain back to the points
    for i in range(len(domain)):
        if i in constant_domain:
            corner_points = np.insert(
                corner_points, i, constant_domain[i], axis=1
            )
            center_points = np.insert(
                center_points, i, constant_domain[i], axis=1
            )

    # Find the edges among corners
    # build a temp ball tree for corner_points
    tree = BallTree(corner_points, metric="euclidean")
    corner_neighbors = [
        tree.query_radius([point], spacing * 1.01)[0]
        for point in corner_points
    ]
    corner_edges = [
        [i, j]
        for i, neighbors in enumerate(corner_neighbors)
        for j in neighbors
        if i < j  # avoid duplication
    ]

    # Find the edges from each center to corresponding corners
    # query the nearest neighbor within spacing for center_points
    center_neighbors = [
        tree.query_radius([point], spacing)[0] for point in center_points
    ]
    # center points are indexed after corner points
    n_corners = len(corner_points)
    center_edges = [
        [i + n_corners, j]
        for i, neighbor in enumerate(center_neighbors)
        for j in neighbor
    ]

    points = np.vstack([corner_points, center_points])
    edges = corner_edges + center_edges
    return np.array(points), np.array(edges)


def get_so3_grid(n_points, rot_domain, fixed_rotation, num_neighbors):
    """Get n_points in SO(3) points using uniform sampling (non-random)

    For one dimension, simply sample n_points in the range of [-pi, pi)
    For three dimensions, sample n_points in a 3-sphere

    Args:
        n_points: number of points to sample
        rot_domain: rot_domain of the rotation to sample, unlike R^3,
                    the rot_domain simply include 1 or 0, indicating
                    whether the rotation is allowed to rotate around
        fixed_rotation: defined in euler angle form (x, y, z)
                        the fixed rotation for the robot, when the rot_domain
                        is not [1, 1, 1], the "0" part of the rotation would be
                        assigned with the fixed rotation
        num_neighbors: number of neighbors for each point
    Returns:
        points: a list of workspace points (coordinates)
        edges: a list of edges in index form
    """
    # Keep a record of the domain that is constant
    num_domian = np.sum(rot_domain)

    # None: no domain is specified, return None
    if num_domian == 0:
        raise ValueError("No domain is specified for rotation grid")

    # Only one angle:
    # simply sample uniformly from -pi to pi, make other angles 0,
    # and connect them in order
    elif num_domian == 1:
        # since angle is cyclic, pi and -pi are the same
        # sample evenly in the range of [-pi, pi)
        angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)

        # create the points with zero for other angles
        eulers = np.tile(fixed_rotation, (n_points, 1))
        index = rot_domain.index(1)
        eulers[:, index] = angles
        quats = [euler_to_quat(euler) for euler in eulers]

    # Only two angle:
    elif num_domian == 2:
        # Compute how many points should be assigned to each dimension
        # n_points_dim = round(n_points ** (1 / 2))
        # TODO
        raise NotImplementedError("Not implemented yet")

    # All SO(3) domain:
    elif num_domian == 3:
        # Evenly distribute points in a 3-sphere represeting quaternion
        # From Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3)
        # https://marcalexa.github.io/superfibonacci/
        # magic number phi and psi
        phi = np.sqrt(2)
        psi = 1.533751168755204288118041

        quats = []
        for i in range(n_points):
            s = i + 0.5
            r1 = np.sqrt(s / n_points)
            r2 = np.sqrt(1 - s / n_points)
            alpha = (2 * np.pi * s) / phi
            beta = (2 * np.pi * s) / psi
            q = (
                r1 * np.sin(alpha),
                r1 * np.cos(alpha),
                r2 * np.sin(beta),
                r2 * np.cos(beta),
            )
            quats.append(q)

    # Find the edges
    # build a temp NN structure for quats
    tree = BallTree(quats, metric=quaternion_angle)
    edges = []
    # + 1 to accomodate self
    neighbors = tree.query(quats, num_neighbors + 1)[1][:, 1:]
    for i, neighbor in enumerate(neighbors):
        for j in neighbor:
            # avoid duplication
            if i < j:
                edges.append([i, j])

    return np.array(quats), np.array(edges)

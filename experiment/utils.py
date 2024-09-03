import numpy as np
from grr.utils import quaternion_angle, matrix_to_quat


def random_unit_vector():
    """Get a random unit vector"""
    vec = np.random.normal(0, 1, 3)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return random_unit_vector()
    return vec / norm


def conf_traj_length(traj, robot):
    """Get the length of the configuration trajectory"""
    length = 0
    for i in range(len(traj) - 1):
        length += robot.distance(traj[i], traj[i + 1])
    return length


def ws_traj_length(traj, robot):
    """Get the length of the work space trajectory"""
    length = 0
    for i in range(len(traj) - 1):
        length += robot.workspace_distance(traj[i], traj[i + 1])
    return length


def check_c_traj(goal, robot, c_traj, num_div=8):
    """Check the configuration trajectory validity"""
    # Check c_traj length
    if len(c_traj) == 0:
        return False

    # Check path validity - check if the end goal is reached
    pos = goal[:3]
    rot = [0, 0, 0, 1]
    if robot.rotation != "free":
        rot = matrix_to_quat(robot.fixed_rotation)

    pose_pos, pose_rot = robot.solve_fk(c_traj[-1], [-1])
    if np.linalg.norm(pose_pos[0] - pos) > 0.1 or (
        robot.rotation != "free" and quaternion_angle(pose_rot[0], rot) > 0.1
    ):
        print("Goal did not reach")
        return False

    # Check path validity - check for self collision
    for i, _ in enumerate(c_traj):
        # interpolate between this and the last one
        # to get a more precise actual workspace trajectory
        if i == 0:
            continue
        for div in range(num_div):
            u = (div + 1) / num_div
            q = robot.interpolate(c_traj[i - 1], c_traj[i], u)
            if robot.check_self_collision(q):
                print("Self collision detected")
                return False

    # Pass test
    return True


def get_ws_traj(start, robot, c_traj, num_div=4):
    """Get the workspace trajectory from the configuration trajectory"""
    w_traj = [start]
    # Get the actual workspace trajectory
    for i, _ in enumerate(c_traj):
        # Interpolate between the last one
        # to get a more precise actual workspace trajectory
        if i == 0:
            continue
        for div in range(num_div):
            u = (div + 1) / num_div
            q = robot.interpolate(c_traj[i - 1], c_traj[i], u)
            pos, rot = robot.solve_fk(q, [-1])
            if robot.rotation == "variable":
                point = np.concatenate([pos[0], rot[0]])
            else:
                point = pos[0]
            w_traj.append(point)
    return w_traj


def dynamic_time_warping(traj1, traj2, distance_func, normalize_over="1"):
    """Dynamic Time Warping"""
    # Init with inf
    dtw_matrix = np.zeros((len(traj1), len(traj2)))
    dtw_matrix.fill(np.inf)

    # Fill the first row and column
    dtw_matrix[0, 0] = 0
    # Fill the other rows and columns
    for i in range(1, len(traj1)):
        for j in range(1, len(traj2)):
            cost = distance_func(traj1[i], traj2[j])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    # Get the index pairs
    i = len(traj1) - 1
    j = len(traj2) - 1
    index_pairs = []
    while i > 0 and j > 0:
        index_pairs.append((i, j))
        index = np.argmin(
            [
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            ]
        )
        if index == 0:
            i -= 1
        elif index == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    # Pair the rest
    if i == 0:
        for jj in range(j + 1):
            index_pairs.append((i, jj))
    elif j == 0:
        for ii in range(i + 1):
            index_pairs.append((ii, j))

    # Get the distance based on the index pairs
    distance = 0
    for i, j in index_pairs:
        distance += distance_func(traj1[i], traj2[j])
    if normalize_over == "1":
        distance /= len(traj1)
    elif normalize_over == "2":
        distance /= len(traj2)
    elif normalize_over == "both":
        distance /= len(traj1) + len(traj2)

    return distance, index_pairs

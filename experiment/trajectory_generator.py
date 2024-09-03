"""This module generates trajectory for evaluation"""

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from grr.utils import rotvec_to_quat, quat_to_matrix, interpolate_quat
from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova

from experiment.utils import random_unit_vector


def sample_valid_circle_start_goal(robot, traj_type, num_valid_test=1):
    """Sample one valid start and goal for circle trajectory"""
    assert traj_type in ["circle_random", "circle_out"]
    trials = 100
    domain_size = min((b - a) for a, b in robot.domain)
    diameter_range_threshold = 0.2 * domain_size

    # Random
    if traj_type == "circle_random":
        while True:
            valid_start, valid_goal = False, False
            start = robot.workspace_sample()
            rand_dist = np.random.uniform(0, diameter_range_threshold)
            goal = robot.workspace_sample()
            goal = np.concatenate(
                [start[:3] + rand_dist * random_unit_vector(), goal[3:]]
            )

            for _ in range(num_valid_test):
                if robot.solve_ik(start) is not None:
                    valid_start = True
                    break
            for _ in range(num_valid_test):
                if robot.solve_ik(goal) is not None:
                    valid_goal = True
                    break

            if valid_start and valid_goal:
                if (
                    np.linalg.norm(start[:3] - goal[:3])
                    < diameter_range_threshold
                ):
                    return start, goal

            trials -= 1
            if trials == 0:
                return None, None

    elif traj_type == "circle_out":
        while True:

            valid_start, valid_goal = False, False
            start = robot.workspace_sample()
            rand_dist = np.random.uniform(0, diameter_range_threshold)
            goal = robot.workspace_sample()
            goal = np.concatenate(
                [start[:3] + rand_dist * random_unit_vector(), goal[3:]]
            )

            for _ in range(num_valid_test):
                if robot.solve_ik(start) is not None:
                    valid_start = True
                    break
            for _ in range(num_valid_test):
                if robot.solve_ik(goal) is not None:
                    valid_goal = True
                    break

            if valid_start and not valid_goal:
                if (
                    np.linalg.norm(start[:3] - goal[:3])
                    < diameter_range_threshold
                ):
                    return start, goal
            elif not valid_start and valid_goal:
                if (
                    np.linalg.norm(start[:3] - goal[:3])
                    < diameter_range_threshold
                ):
                    return goal, start

            trials -= 1
            if trials == 0:
                return None, None


def sample_valid_line_start_goal(robot, traj_type, num_valid_test=1):
    """Sample one valid start and goal for line trajectory"""
    assert traj_type in ["line_random", "line_self"]
    trials = 100

    # Random
    if traj_type == "line_random":
        while True:

            valid_start, valid_goal = False, False
            start = robot.workspace_sample()
            goal = robot.workspace_sample()

            for _ in range(num_valid_test):
                if robot.solve_ik(start) is not None:
                    valid_start = True
                    break
            for _ in range(num_valid_test):
                if robot.solve_ik(goal) is not None:
                    valid_goal = True
                    break
            if valid_start and valid_goal:
                return start, goal

            trials -= 1
            if trials == 0:
                return None, None

    # Self-crossing
    elif traj_type == "line_self":
        while True:

            valid_start, valid_goal = False, False
            start = robot.workspace_sample()
            for _ in range(num_valid_test):
                if robot.solve_ik(start) is not None:
                    valid_start = True
                    break

            if valid_start:
                # Use the opposite point as self-crossing goal
                goal = -start[:3]
                # TODO use opposite rotation
                # if rotation included, use the same rotation
                if len(start) > 3:
                    goal = np.concatenate([goal, start[3:]])

                for _ in range(num_valid_test):
                    if robot.solve_ik(goal) is not None:
                        valid_goal = True
                        break

            if valid_start and valid_goal:
                return start, goal

            trials -= 1
            if trials == 0:
                return None, None


def generate_trajectory(robot, num_traj, inter_time, inter_hz, traj_type):
    """Randomly generate start and goal, and interpolate the path"""
    assert traj_type in [
        "line_random",
        "line_self",
        "circle_random",
        "circle_out",
    ]

    trajectories = []

    if traj_type == "line_random" or traj_type == "line_self":
        pbar = tqdm(total=num_traj)
        while len(trajectories) < num_traj:
            # Randomly sample valid start and goal
            start, goal = sample_valid_line_start_goal(robot, traj_type)
            if start is None or goal is None:
                continue
            pbar.update(1)

            # Interpolate using straight line:
            # Ensure at least start and goal are included
            num_points = max(int(inter_time * inter_hz), 1)

            # Interpolate the path
            path = []
            for point_idx in range(num_points + 1):
                time_t = point_idx / num_points
                # Get the points on the line by interpolation
                u = point_idx / num_points
                point = robot.workspace_interpolate(start, goal, u)
                path.append((time_t, point))

            trajectories.append(path)
        pbar.close()

    if traj_type == "circle_random" or traj_type == "circle_out":
        pbar = tqdm(total=num_traj)
        while len(trajectories) < num_traj:
            # Randomly sample valid start and goal
            start, goal = sample_valid_circle_start_goal(robot, traj_type)
            if start is None or goal is None:
                continue
            pbar.update(1)

            # Interpolate using a circle:
            # Ensure at least start and goal are included
            num_points = max(int(inter_time * inter_hz), 1)

            # Interpolate the circle
            center = (np.array(start[:3]) + np.array(goal[:3])) / 2
            diameter_dir = goal[:3] - start[:3]
            # endpoint=True -> close the loop
            angles = np.linspace(0, 2 * np.pi, num_points + 1, endpoint=True)

            # decide the plane of the circle
            up_vector = random_unit_vector()
            while np.isclose(
                np.dot(up_vector, diameter_dir / np.linalg.norm(diameter_dir)),
                1.0,
            ):
                up_vector = random_unit_vector()
            base_dir = np.cross(diameter_dir, up_vector)
            base_dir /= np.linalg.norm(base_dir)

            # Get the points on the circle
            path = []
            for point_idx in range(num_points + 1):
                time_t = point_idx / num_points

                # rotation matrix
                rotation = quat_to_matrix(
                    rotvec_to_quat(base_dir * angles[point_idx])
                )
                # rotate base_dir to get point direction
                point = center + rotation.dot(start[:3] - center)

                # circle rotation if needed
                if len(start) > 3:
                    # first half: start -> goal, second half: goal -> start
                    u = 2 * point_idx / num_points
                    if u > 1:
                        u = 2 - u
                    rot = interpolate_quat(start[3:], goal[3:], u)
                    point = np.concatenate([point, rot])

                path.append((time_t, point))
            # closed loop
            path.append((1.0, path[0][1]))

            trajectories.append(path)
        pbar.close()

    return trajectories


def main(opts):
    """Main function to run the demo"""
    RobotClass = getattr(sys.modules[__name__], opts["robot_class"])
    robot = RobotClass(
        opts["robot_name"],
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # Graph folder path
    pardir = os.path.dirname(os.path.dirname(__file__))
    trajectory_folder = pardir + (
        "/experiment/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    if not os.path.exists(trajectory_folder):
        os.makedirs(trajectory_folder)

    ### Trajectory for teleoperation experiment - line ###

    # Generate the trajectory for line - random
    print("Generating trajectory for teleoperation - random")
    trajectories = generate_trajectory(
        robot, num_traj=100, inter_time=4, inter_hz=50, traj_type="line_random"
    )
    # Save trajectory
    with open(trajectory_folder + "line_random.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    # Generate the trajectory for line - self-crossing
    print("Generating trajectory for teleoperation - self-crossing")
    trajectories = generate_trajectory(
        robot, num_traj=100, inter_time=4, inter_hz=50, traj_type="line_self"
    )
    # Save trajectory
    with open(trajectory_folder + "line_self.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    ### Trajectory for teleoperation experiment - circle ###

    # Generate the trajectory for circle - random
    print("Generating trajectory for teleoperation - circle random")
    trajectories = generate_trajectory(
        robot,
        num_traj=100,
        inter_time=4,
        inter_hz=50,
        traj_type="circle_random",
    )
    # Save trajectory
    with open(trajectory_folder + "circle_random.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    # Generate the trajectory for circle - out
    print("Generating trajectory for teleoperation - circle out")
    trajectories = generate_trajectory(
        robot, num_traj=100, inter_time=4, inter_hz=50, traj_type="circle_out"
    )
    # Save trajectory
    with open(trajectory_folder + "circle_out.pkl", "wb") as f:
        pickle.dump(trajectories, f)


def visualize_path(start, goal, trajectory):
    """Visualize the generated trajectory"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot start and goal
    start = trajectory[0][1]
    goal = trajectory[-1][1]
    ax.scatter(start[0], start[1], start[2], c="r", marker="o")
    ax.scatter(goal[0], goal[1], goal[2], c="g", marker="o")

    # Plot path
    path = np.array([p[1] for p in trajectory])
    ax.plot(path[:, 0], path[:, 1], path[:, 2], c="b")

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    plt.show()


if __name__ == "__main__":
    # Default json file
    robot_name = "kinova"
    json_file_name = "rot_fixed"

    # Override with system arguments if provided
    if len(sys.argv) == 2:
        print("Need to specify both the robot and the json file")
        print("python demo.py <robot> <json>")
    elif len(sys.argv) > 2:
        robot_name = sys.argv[1]
        json_file_name = sys.argv[2]

    opts = load_json(robot_name, json_file_name)
    main(opts)

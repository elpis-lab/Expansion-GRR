"""This module evaluate the trajectory quality"""

import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw

from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova
from grr.resolution import RedundancyResolution
from grr.utils import matrix_to_quat
from experiment.relaxed_ik_core.relaxed_ik import RelaxedIKRust

from experiment.utils import (
    conf_traj_length,
    ws_traj_length,
    check_c_traj,
    get_ws_traj,
)


def newton_teleop_solver(traj, resolution):
    """Newton-Raphson IK solver"""
    c_traj = []

    # Init config with the config provided by resolution
    config = resolution.solve(traj[0][1])
    # not valid to start with this
    if config is None:
        print("Start point is not valid")
        return []
    if resolution.solve(traj[-1][1]) is None:
        print("End point is not valid")
        return []

    # Get the configuration trajectory
    for i, (_, point) in enumerate(traj):
        # config waypoint
        res = resolution.solve(point, config, regular_ik=True)
        if res is not None:
            res = resolution.teleop_towards(config, res, max_change=0.04)
        c_traj.append(res)
        if c_traj[-1] is not None:
            config = c_traj[-1]

    # At the end of the path, try to run GRR multiple times
    # to see if GRR can converge
    point = traj[-1][1]
    for i in range(100):
        res = resolution.solve(point, config, regular_ik=True)
        if res is not None:
            res = resolution.teleop_towards(config, res, max_change=0.04)
        c_traj.append(res)
        if c_traj[-1] is not None:
            config = c_traj[-1]

    # If None provided, self collision is detected on the way
    # temporarily skip it
    for i, config in enumerate(c_traj):
        if config is None:
            c_traj[i] = c_traj[i - 1]

    return c_traj


def grr_teleop_solver(traj, resolution):
    """Teleoperation solver"""
    c_traj = []

    # Init config with the config provided by resolution
    config = resolution.solve(traj[0][1])
    # not valid to start with this
    if config is None:
        print("Start point is not valid")
        return []
    if resolution.solve(traj[-1][1]) is None:
        print("End point is not valid")
        return []

    # Get the configuration trajectory
    # switch to teleop
    resolution.planning_mode = False
    for i, (_, point) in enumerate(traj):
        # config waypoint
        res = resolution.teleop_solve(point, config, max_change=0.04)
        c_traj.append(res)
        if c_traj[-1] is not None:
            config = c_traj[-1]

    # At the end of the path, try to run GRR multiple times
    # to see if GRR can converge
    point = traj[-1][1]
    for i in range(100):
        res = resolution.teleop_solve(point, config, max_change=0.04)
        c_traj.append(res)
        if c_traj[-1] is not None:
            config = c_traj[-1]

    # If None provided, self collision is detected on the way
    # temporarily skip it
    for i, config in enumerate(c_traj):
        if config is None:
            c_traj[i] = c_traj[i - 1]

    return c_traj


def relaxed_ik_teleop_solver(relaxed_ik, traj, resolution):
    """Teleoperation solver"""
    c_traj = []
    robot = resolution.robot

    # Init config with the config provided by resolution
    config = resolution.solve(traj[0][1])
    # not valid to start with this
    if config is None:
        print("Start point is not valid")
        return []
    if resolution.solve(traj[-1][1]) is None:
        print("End point is not valid")
        return []

    # Get the configuration trajectory
    relaxed_ik.reset(config)
    for i, (_, point) in enumerate(traj):
        pos = point[:3]
        # TODO relaxed-ik does not support position only now
        rot = [0, 0, 0, 1]
        if robot.rotation == "variable":
            rot = point[3:7]
        elif robot.rotation == "fixed":
            rot = matrix_to_quat(robot.fixed_rotation)

        res = relaxed_ik.solve(pos, rot)
        c_traj.append(res)

    # At the end of the path, try to run relaxed-IK multiple times
    # to see if relaxed-ik can converge
    res = relaxed_ik.solve_precise(pos, rot, 100)
    c_traj.append(res)

    return c_traj


def run_teleoperation(trajectories, solver, save_name):
    """Run the teleoperation experiment and return the solution"""
    print("Running teleoperation test")
    # Storage
    c_trajs = []
    i_trajs = []

    # Run the motion planning experiment
    for _, t_traj in enumerate(tqdm(trajectories)):
        # Plan
        c_traj = solver(t_traj)  # input: list of (t, point)
        # Save results
        c_trajs.append(c_traj)
        i_trajs.append(np.array([t[1] for t in t_traj]))

    # Save the results
    pardir = os.path.dirname(os.path.dirname(__file__))
    with open(
        pardir + "/experiment/results/" + save_name + "_c_trajs.pkl", "wb"
    ) as f:
        pickle.dump((c_trajs, i_trajs), f)
    return c_trajs


def run_experiment_grr(trajectories, resolution, save_name):
    """Run the experiment with global redundancy resolution"""

    # Run teleoperation test
    def grr_solver(traj):
        return grr_teleop_solver(traj, resolution)

    return run_teleoperation(trajectories, grr_solver, save_name)


def run_experiment_newton_ik(trajectories, resolution, save_name):
    """Run the experiment with newton raphson IK"""

    # Run teleoperation test
    def newton_solver(traj):
        return newton_teleop_solver(traj, resolution)

    return run_teleoperation(trajectories, newton_solver, save_name)


def run_experiment_relaxed_ik(trajectories, resolution, save_name):
    """Run the experiment with relaxed IK"""
    # Run teleoperation test
    relaxed_ik = RelaxedIKRust()

    def relaxed_solver(traj):
        return relaxed_ik_teleop_solver(relaxed_ik, traj, resolution)

    return run_teleoperation(trajectories, relaxed_solver, save_name)


def analyze_results(
    trajectories, robot, c_trajs_collection, only_all_success=True
):
    """Analyzing the results of the experiment"""
    # Storage
    results = []
    failed_indices = [set() for _ in range(len(c_trajs_collection))]

    # Analyze the validity first
    for i, c_trajs in enumerate(c_trajs_collection):
        print("\nAnalyzing the validity results - ", i)
        for j, traj in enumerate(tqdm(trajectories)):
            c_traj = c_trajs[j]

            # Check solution validity
            validity = check_c_traj(traj[-1][1], robot, c_traj)
            if not validity:
                failed_indices[i].add(j)

    # If we only compare the successful ones
    failed = set()
    if only_all_success:
        for _, failed_index in enumerate(failed_indices):
            failed.update(failed_index)
        print("Failed indices count", len(failed))
        print("Failed indices combined", failed)

    # Get the results
    all_results = []
    for i, c_trajs in enumerate(c_trajs_collection):
        print("\nAnalyzing the value results - ", i)

        results = []
        for j, traj in enumerate(tqdm(trajectories)):
            if only_all_success and j in failed:
                continue
            if j in failed_indices[i]:
                continue
            c_traj = c_trajs[j]

            # Compute
            w_traj = get_ws_traj(traj[0][1], robot, c_traj)
            c_length = conf_traj_length(c_traj, robot)
            w_length = ws_traj_length(w_traj, robot)
            i_traj = np.array([t[1] for t in traj])
            i_length = ws_traj_length(i_traj, robot)
            deviation = fastdtw(i_traj, w_traj, dist=robot.workspace_distance)[
                0
            ]
            norm_deviation = deviation / len(i_traj)

            results.append((c_length, w_length, i_length, norm_deviation))

        # Get the results for this c_trajs
        if results == []:
            print("No valid solution found")
            all_results.append([0, 0, 0, 0, 0])
            continue

        # Analyze the results
        results = np.atleast_2d(results)
        c_lengths = results[:, 0]
        w_lengths = results[:, 1]
        i_lengths = results[:, 2]
        deviation = results[:, 3]

        dev = np.mean(deviation)
        c_len, w_len = np.mean(c_lengths), np.mean(w_lengths)
        dist_ratio = np.mean(c_lengths / w_lengths)
        success_rate = (1 - len(failed_indices[i]) / len(trajectories)) * 100

        print("Failed indices", failed_indices[i])
        print("Deviation From input", dev)
        print("Configuration Trajectory Length", c_len)
        print("Workspace Trajectory Length", w_len)
        print("Distance Ratio", dist_ratio)
        print("Success Rate", success_rate, "%")
        # all_results.append([dev, c_len, w_len, dist_ratio, success_rate])
        all_results.append([dev, dist_ratio, success_rate])

    print()
    for result in all_results:
        print(result)
    return all_results


def main(opts):
    """Main function to test the quality of trajectories"""
    RobotClass = getattr(sys.modules[__name__], opts["robot_class"])
    robot = RobotClass(
        opts["robot_name"],
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # Load test trajectories
    pardir = os.path.dirname(os.path.dirname(__file__))
    trajectory_folder = pardir + (
        "/experiment/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    experiment_files = [
        "line_random",
        "line_self",
        "circle_random",
        "circle_out",
    ]

    # Storage for the results
    results = [[] for _ in range(len(experiment_files))]

    # Run test
    for _, experiment_file in enumerate(experiment_files):
        print("\nStart testing - ", experiment_file)
        with open(trajectory_folder + experiment_file + ".pkl", "rb") as f:
            trajectories = pickle.load(f)

        # Load testings - ExpansionGRR
        graph_folder = pardir + (
            "/graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
        )
        # Run testings - ExpansionGRR
        egrr_resolution = RedundancyResolution(robot)
        egrr_resolution.load_resolution_graph(
            graph_folder + "graph_resolution.pickle",
            graph_folder + "nn_resolution.pickle",
        )
        print("\nRunning Expansion GRR")
        egrr_c_trajs = run_experiment_grr(
            trajectories,
            egrr_resolution,
            save_name=experiment_file + "_egrr",
        )

        # Load testings - RandomGRR
        graph_folder = pardir + (
            "/experiment/rgrr/"
            + opts["robot_name"]
            + "/"
            + opts["problem_type"]
            + "/"
        )
        # Run testings - RandomGRR
        rgrr_resolution = RedundancyResolution(robot)
        rgrr_resolution.load_resolution_graph(
            graph_folder + "graph_resolution.pickle",
            graph_folder + "nn_resolution.pickle",
        )
        print("\nRunning Random GRR")
        rgrr_c_trajs = run_experiment_grr(
            trajectories,
            rgrr_resolution,
            save_name=experiment_file + "_rgrr",
        )

        # Run testings - Newton-Raphson IK
        print("\nRunning Newton-Raphson IK")
        newton_c_trajs = run_experiment_newton_ik(
            trajectories,
            egrr_resolution,
            save_name=experiment_file + "_newton",
        )

        # Run testings - Relaxed IK
        print("\nRunning Relaxed IK")
        relax_c_trajs = run_experiment_relaxed_ik(
            trajectories,
            egrr_resolution,
            save_name=experiment_file + "_relax",
        )

        result_dir = pardir + "/experiment/results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # Load the trajectory with pickle
        with open(
            result_dir + "/" + experiment_file + "_egrr_c_trajs.pkl", "rb"
        ) as f:
            egrr_c_trajs, w_trajs = pickle.load(f)
        # with open(
        #     result_dir + "/" + experiment_file + "_rgrr_c_trajs.pkl", "rb"
        # ) as f:
        #     rgrr_c_trajs, w_trajs = pickle.load(f)
        # with open(
        #     result_dir + "/" + experiment_file + "_newton_c_trajs.pkl", "rb"
        # ) as f:
        #     newton_c_trajs, w_trajs = pickle.load(f)
        # with open(
        #     result_dir + "/" + experiment_file + "_relax_c_trajs.pkl", "rb"
        # ) as f:
        #     relax_c_trajs, w_trajs = pickle.load(f)

        print("\nAnalyzing the results")
        result = analyze_results(
            trajectories,
            robot,
            [egrr_c_trajs],  # rgrr_c_trajs, newton_c_trajs, relax_c_trajs],
            only_all_success=True,
        )
        results.append(result)

    # Print the results
    print("\nResults - ")
    for r1 in results:
        for r2 in r1:
            print(r2)


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

"""This module visualize the trajectory"""

import sys
import os
import pickle

from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova
from grr.resolution import RedundancyResolution
from visualization.klampt_vis import GLRedundancyProgram


def main(opts):
    """Main function to test the quality of trajectories"""
    RobotClass = getattr(sys.modules[__name__], opts["robot_class"])
    robot = RobotClass(
        opts["robot_name"],
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # Graph folder path
    pardir = os.path.dirname(os.path.dirname(__file__))
    graph_folder = pardir + (
        "/graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    result_dir = pardir + "/experiment/results"

    # Global redundancy resolution
    resolution = RedundancyResolution(robot)
    # resolution.load_solver_graph(graph_folder + "graph_solver.pickle")
    # resolution.load_resolution_graph(
    #     graph_folder + "graph_resolution.pickle",
    #     graph_folder + "nn_resolution.pickle",
    # )

    # Run the demonstration
    program = GLRedundancyProgram(resolution, width=720, height=720)

    # Load the trajectory with pickle
    with open(result_dir + "/line_random_egrr_c_trajs.pkl", "rb") as f:
        egrr_c_trajs, w_trajs = pickle.load(f)
    with open(result_dir + "/line_random_rgrr_c_trajs.pkl", "rb") as f:
        rgrr_c_trajs, w_trajs = pickle.load(f)
    with open(result_dir + "/line_random_newton_c_trajs.pkl", "rb") as f:
        newton_c_trajs, w_trajs = pickle.load(f)
    with open(result_dir + "/line_random_relax_c_trajs.pkl", "rb") as f:
        relax_c_trajs, w_trajs = pickle.load(f)

    # Index
    index = 0
    solution_index = 1
    c_trajs = [egrr_c_trajs, rgrr_c_trajs, newton_c_trajs, relax_c_trajs]

    program.set_workspace_path(w_trajs[index])
    program.set_path(c_trajs[solution_index][index])
    program.run()
    exit(0)


if __name__ == "__main__":
    # Default json file
    robot_name = "kinova"
    json_file_name = "rot_free"

    # Override with system arguments if provided
    if len(sys.argv) == 2:
        print("Need to specify both the robot and the json file")
        print("python demo.py <robot> <json>")
    elif len(sys.argv) > 2:
        robot_name = sys.argv[1]
        json_file_name = sys.argv[2]

    opts = load_json(robot_name, json_file_name)
    main(opts)

"""Main file to run global redundancy resolution to find a solution for a robot"""

import os
import sys
import time

from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova
from grr.resolution import RedundancyResolution

from experiment.roadmap_quality import evaluate_roadmap


def main(opts, load_existed_ws_graph=False, load_existed_solver_graph=False):
    """Run global redundancy resolution"""
    RobotClass = getattr(sys.modules[__name__], opts["robot_class"])
    robot = RobotClass(
        opts["robot_name"],
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # Graph folder path
    graph_folder = (
        "graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    # Global redundancy resolution
    resolution = RedundancyResolution(robot)

    # Build workspace graph
    if not load_existed_ws_graph:
        resolution.sample_workspace(
            n_pos_points=opts["number_of_position_points"],
            n_rot_points=opts["number_of_rotation_points"],
            sampling_method="grid",
        )
        resolution.save_workspace_graph(
            graph_folder + "graph_workspace.pickle",
            graph_folder + "nn_workspace.pickle",
        )
    else:
        resolution.load_workspace_graph(
            graph_folder + "graph_workspace.pickle",
            graph_folder + "nn_workspace.pickle",
        )
    # resolution.visualize_workspace_graph()

    # Timer
    start_time = time.time()

    # Build configuration space graph
    if not load_existed_solver_graph:
        resolution.global_expansion(opts["init_configs"])
        resolution.save_solver_graph(graph_folder + "graph_solver.pickle")
    else:
        resolution.load_solver_graph(graph_folder + "graph_solver.pickle")

    # Optimization
    resolution.fix_boundary(n_neighbor_layer=1, n_iter=2)
    # TODO Future Improvement
    # resolution.optimize_resolution(n_length_optimization=2)

    # Timer
    print("Total Computation Time:", time.time() - start_time)

    # Build resolution graph
    resolution.build_resolution_graph_and_nn()
    resolution.save_solver_graph(graph_folder + "graph_solver.pickle")
    resolution.save_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )
    # resolution.load_resolution_graph(
    #     graph_folder + "graph_resolution.pickle",
    #     graph_folder + "nn_resolution.pickle",
    # )

    # Evaluate the roadmap
    evaluate_roadmap(resolution)


if __name__ == "__main__":
    # Default json file
    robot_name = "planar_5"
    json_file_name = "rot_free"

    # Override with system arguments if provided
    if len(sys.argv) == 2:
        print("Need to specify both the robot and the json file")
        print("python redundancy.py <robot> <json>")
    elif len(sys.argv) > 2:
        robot_name = sys.argv[1]
        json_file_name = sys.argv[2]

    opts = load_json(robot_name, json_file_name)
    main(opts, False, False)

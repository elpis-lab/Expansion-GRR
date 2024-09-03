"""This module evaluate the built roadmap quality"""

import sys
import os
import numpy as np

from grr.json_utils import load_json
from grr.robot import KinematicChain, Kinova
from grr.resolution import RedundancyResolution


def evaluate_roadmap(resolution):
    """Evaluate the roadmap quality"""
    # Evaluate the roadmap quality
    graph = resolution.solver.graph
    robot = resolution.robot
    print("\nStart to evaluate the roadmap quality")
    print("Number of nodes:", len(graph.nodes))
    print("Number of edges:", len(graph.edges))

    # Evaluate the resolution quality
    # 1, Disconnection Ratio
    num_edges = 0
    num_disconnected = 0
    for i, j, edge in graph.edges(data=True):
        if (
            graph.nodes[i]["config"] is None
            or graph.nodes[j]["config"] is None
        ):
            continue

        num_edges += 1
        if not edge["connected"]:
            num_disconnected += 1
    print("Disconnection Ratio:", num_disconnected / num_edges * 100, "%")

    # 2, Distance ratio
    dist_ratio = []
    for i, j, edge in graph.edges(data=True):
        if (
            graph.nodes[i]["config"] is None
            or graph.nodes[j]["config"] is None
        ):
            continue

        config1 = graph.nodes[i]["config"]
        config2 = graph.nodes[j]["config"]
        point1 = graph.nodes[i]["point"]
        point2 = graph.nodes[j]["point"]
        dist_ratio.append(
            robot.distance(config1, config2)
            / robot.workspace_distance(point1, point2)
        )
    print("Distance Ratio:", np.mean(dist_ratio), "rad/m")


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
    graph_folder = pardir + (
        "/graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )

    # Global redundancy resolution - Expansion GRR
    print("Evaluating the roadmap quality - Expansion GRR")
    resolution = RedundancyResolution(robot)
    resolution.load_solver_graph(graph_folder + "graph_solver.pickle")
    resolution.load_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )

    # Run the evaluation
    evaluate_roadmap(resolution)

    # Global redundancy resolution - Random GRR
    print("Evaluating the roadmap quality - Random GRR")
    graph_folder = pardir + (
        "/experiment/rgrr/"
        + opts["robot_name"]
        + "/"
        + opts["problem_type"]
        + "/"
    )
    resolution = RedundancyResolution(robot)
    resolution.load_solver_graph(graph_folder + "graph_solver.pickle")
    resolution.load_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )

    # Run the evaluation
    evaluate_roadmap(resolution)


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

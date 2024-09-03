"""This module demonstrates the robot resolution with Ursina"""

import sys
from grr.json_utils import load_json

from grr.robot import KinematicChain, Kinova
from grr.resolution import RedundancyResolution
from visualization.klampt_vis import GLRedundancyProgram


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
    graph_folder = (
        "graph/" + opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )

    # Global redundancy resolution
    resolution = RedundancyResolution(robot)
    resolution.load_solver_graph(graph_folder + "graph_solver.pickle")
    resolution.load_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )

    # Run the demonstration
    program = GLRedundancyProgram(resolution, width=720, height=720)
    program.run()
    exit(0)


if __name__ == "__main__":
    # Default json file
    robot_name = "planar_5"
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

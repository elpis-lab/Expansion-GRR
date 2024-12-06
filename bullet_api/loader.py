"""Load GRR roadmap for a robot and problem type."""

import os
import sys

from ..grr.resolution import RedundancyResolution
from ..grr.json_utils import load_json
from .robot import KinematicChain, Kinova, UR10

# Needed for loading pickled files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_grr(robot_name, roadmap_type):
    # Dictionary mapping for robot classes
    robot_classes = {
        "KinematicChain": KinematicChain,
        "Kinova": Kinova,
        "UR10": UR10,
    }
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf = os.path.join(root_dir, "data", "robots", robot_name + ".urdf")

    # Load GRR
    opts = load_json(robot_name, roadmap_type)

    # grr robot
    RobotClass = robot_classes[opts["robot_class"]]
    grr_robot = RobotClass(
        urdf,
        opts["domain"],
        opts["rotation_domain"],
        opts["fixed_rotation"],
    )

    # grr graph
    grr = RedundancyResolution(grr_robot)

    dir_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_folder = (dir_folder + "/graph/") + (
        opts["robot_name"] + "/" + opts["problem_type"] + "/"
    )
    # grr.load_solver_graph(graph_folder + "graph_solver.pickle")
    grr.load_resolution_graph(
        graph_folder + "graph_resolution.pickle",
        graph_folder + "nn_resolution.pickle",
    )

    return grr

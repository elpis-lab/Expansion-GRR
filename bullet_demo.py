"""A demo of the Bullet API."""

import os
import numpy as np
from bullet_api.loader import load_grr


def main():
    """Build a demo"""
    # Load a GRR resolution
    grr = load_grr(
        "data/robots/ur10_robotis_d435.urdf",
        "ur10",
        "rot_variable_yaw",
    )

    print(grr.solve([0.5, 0, 0, 0, 0, 0, 1], none_on_fail=True))

    # TODO for the users
    # Define workspace path in the robot base frame
    workspace_path = [
        ([0.5, -0.3, 0.05], [0.71, 0.71, 0, 0]),
        ([0.5, -0.7, 0.05], [0.7071068, 0.7071068, 0, 0]),
    ]

    # Solve the Cartesian path with GRR
    config_path = grr_plan(grr, workspace_path)

    # If you need to build a trajectory,
    # include the time path to generate a trajectory
    time_path = np.linspace(0, 2, len(config_path))
    traj = [(t, q) for t, q in zip(time_path, config_path)]

    for waypoint in traj:
        print(f"Time {waypoint[0]}: {waypoint[1]}")


def grr_plan(grr, workspace_path):
    """Plan pushing with GRR"""
    config_path = [
        grr.solve(np.array([0.5, 0, 0, 0, 0, 0, 1]), none_on_fail=True)
        for waypoint in workspace_path
    ]

    print(workspace_path[0][0] + workspace_path[0][1])
    print(config_path)

    # Valid solution check
    for conf in config_path:
        if conf is None:
            print("\nInvalid configuration found\n")
            return None

    # TODO 1
    # Collision checking with obstacles is not implimented

    # TODO 2
    # Continuity checking is not performed,
    # although this should be guranteed by GRR.
    # One corner case is that base joint believes 0 and 2pi are the same,
    # but the physical robot will not think so.

    return config_path


if __name__ == "__main__":
    main()

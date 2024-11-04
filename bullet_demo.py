"""A demo of the Bullet API."""

import os
import numpy as np
from bullet_api.loader import load_grr


def main():
    """Build a demo"""
    # Load a GRR resolution
    dir = os.path.dirname(os.path.abspath(__file__))
    grr = load_grr(
        dir + "/data/robots/ur10_bullet.urdf",
        "ur10",
        "rot_variable_yaw",
    )

    # TODO for the users
    # Define workspace path in the robot base frame
    workspace_path = [
        ([0.5, -0.25, 0.05], [0.7071068, 0.7071068, 0, 0]),
        ([0.5, 0.0, 0.05], [0.7071068, 0.7071068, 0, 0]),
        ([0.5, 0.25, 0.05], [0.7071068, 0.7071068, 0, 0]),
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
        grr.solve(waypoint[0] + waypoint[1], none_on_fail=True)
        for waypoint in workspace_path
    ]

    # # Debug
    # for config in config_path:
    #     print(config)

    # TODO 0
    # Valid solution check
    for conf in config_path:
        if conf is None:
            print("\nInvalid configuration found\n")
            return config_path

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

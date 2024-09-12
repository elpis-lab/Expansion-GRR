"""Module providing a class that defines a robot"""

import sys
import os
import numpy as np

import klampt
from klampt.model import ik, collide

from .utils import se3_distance
from .utils import wrap_to_pi, interpolate_angle
from .utils import sample_quat, interpolate_quat
from .utils import euler_to_quat, euler_to_matrix
from .utils import quat_to_matrix, matrix_to_quat


class Robot:
    """A meta robot class"""

    def __init__(self, name, domain, rot_domain, fixed_rotation=None):
        """Initialize a robot from a rob file (Klampt),
        given a domain and a rotation type

        Args:
            name: the name of the robot file
            domain: a 3D domain for the position component, defined as
                [min_x, max_x], [min_y, max_y], [min_z, max_z]
            rot_domain: a rotation domain for the rotation component,
                defined as [1/0, 1/0, 1/0].
                For example:
                [0, 0, 0] means rotation is free
                [1, 0, 0] means rotation is variable in x, others are kept 0
                [1, 1, 1] means rotation is variable in x, y and z
            fixed_rotation: a fixed rotation for the rotation component,
                defined as euler angles [q_x, q_y, q_z], in radian
        """
        self.name = name
        self.world = klampt.WorldModel()
        pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.world.loadElement(pardir + "/data/robots/" + name + ".rob")
        self.robot = self.world.robot(0)

        self.domain = domain
        self.rot_domain = rot_domain
        if np.sum(rot_domain) != 0:
            self.rotation = "variable"
            self.fixed_rotation = None
        elif fixed_rotation is not None:
            self.rotation = "fixed"
            # Convert euler to rotation matrix if size is 3
            fixed_rotation = np.array(fixed_rotation)
            if len(fixed_rotation.shape) == 1 and len(fixed_rotation) == 3:
                fixed_rotation = euler_to_matrix(fixed_rotation)
            self.fixed_rotation = fixed_rotation
        else:
            self.rotation = "free"
            self.fixed_rotation = None

        # Initialize the robot attributes
        # This can be customized for different robots
        joint_limits = np.array(self.robot.getJointLimits()).T
        active_joints = self.get_active_joints(joint_limits)
        self.init_attributes(active_joints)
        # assume the last joint of the robot
        self.robot_ee = self.robot.link(self.robot.numLinks() - 1)

    def init_attributes(self, active_joints):
        """Initialize the attributes"""
        self.active_joints = active_joints

        joint_limits = np.array(self.robot.getJointLimits()).T
        self.joint_limits = joint_limits[self.active_joints]
        self.num_joints = len(self.active_joints)
        self.cyclic_joints = self.get_cyclic_joints(self.joint_limits)
        # get only the active ones
        self.links = [self.robot.link(i) for i in self.active_joints]

    def get_active_joints(self, limits):
        """Return the active joint (non-fixed joints)"""
        joints = []
        for i, limit in enumerate(limits):
            if limit[0] != limit[1]:
                joints.append(i)
        return joints

    def get_cyclic_joints(self, joint_limits):
        """Return the cyclic joint indices"""
        return np.array(
            [
                i
                for i, limit in enumerate(joint_limits)
                if limit[0] == -np.inf or limit[1] == np.inf
            ]
        )

    def workspace_sample(self):
        """Samples a point (R^3 or SE3) from the worskapce.

        Return either a
            - 3D point [x, y, z] or
            - 6D point [x, y, z, q_x, q_y, q_z, q_w]
        """
        # Sample in position (R^3)
        point = [np.random.uniform(a, b) for (a, b) in self.domain]

        # Sample in rotation (SO(3))
        # TODO test this for rotation SO(3)
        # variable
        if self.rotation == "variable":
            # if only need to sample in one rotation dimension
            if np.sum(self.rot_domain) == 1:
                angle = np.random.uniform(-np.pi, np.pi)
                euler = np.zeros(3)
                index = self.rot_domain.index(1)
                euler[index] = angle
                quat = euler_to_quat(euler)

            # if need to sample in two rotation dimensions
            elif np.sum(self.rot_domain) == 2:
                # TODO
                raise NotImplementedError

            # if need to sample in so3
            else:
                quat = sample_quat()

            point.extend(quat)

        # free - do not include SO(3) component
        else:
            pass

        return np.array(point)

    def workspace_distance(self, point1, point2):
        """Compute the distance between two workspace points."""
        return se3_distance(np.array(point1), np.array(point2))

    def workspace_interpolate(self, point1, point2, u):
        """Interpolate between two workspace points."""
        point1 = np.array(point1)
        point2 = np.array(point2)

        # Position interpolation
        pos = point1[:3] + u * (point2[:3] - point1[:3])

        # Rotation interpolation
        # TODO test this for rotation SO(3)
        # variable
        if self.rotation == "variable":
            quat = interpolate_quat(point1[3:7], point2[3:7], u)
            point = np.concatenate([pos, quat])

        # free
        else:
            point = pos

        return point

    def sample(self):
        """Sample a configuration in the configuration space"""
        config = np.zeros(len(self.joint_limits))
        for i, limit in enumerate(self.joint_limits):

            # If the joint is cyclic, simply sample from [-pi, pi)
            if i in self.cyclic_joints:
                config[i] = np.random.uniform(-np.pi, np.pi)

            # Otherwise, sample from the joint limit
            else:
                config[i] = np.random.uniform(limit[0], limit[1])

        return np.array(config)

    def distance(self, config1, config2):
        """Compute the distance between two configurations"""
        config1 = np.array(config1)
        config2 = np.array(config2)

        diff = config1 - config2
        # For cyclic joints, wrap the difference to [-pi, pi)
        for i in self.cyclic_joints:
            diff[i] = wrap_to_pi(diff[i])

        return np.linalg.norm(diff)

    def interpolate(self, config1, config2, u):
        """Compute the configuration at a given interpolation parameter"""
        config1 = np.array(config1)
        config2 = np.array(config2)

        config = config1 + u * (config2 - config1)
        # For cyclic joints, interpolate the angle properly
        for i in self.cyclic_joints:
            config[i] = interpolate_angle(config1[i], config2[i], u)
        return config

    def average(self, configs, weights=None):
        """Get an average configuration
        given a set of configurations with weights.
        """
        configs = np.array(configs)

        # If weighs not provided or not valid, use uniform weights
        if weights is None or np.sum(weights) == 0:
            weights = np.ones(len(configs)) / len(configs)

        # Compute the weighted average for non-cyclic joints
        q_res = np.average(configs, axis=0, weights=weights)

        # For cyclic joints, average the angles with circular mean instead
        for i in self.cyclic_joints:
            angles = np.array(configs)[:, i]
            x = np.sum(weights * np.cos(angles))
            y = np.sum(weights * np.sin(angles))
            q_res[i] = np.arctan2(y, x)

        return q_res

    def solve_fk(self, config, index=None):
        """Solve the forward kinematics problem"""
        # Set config
        q = np.array(self.robot.getConfig())
        q[self.active_joints] = config
        self.robot.setConfig(list(q))

        # Acquire positions and rotations
        # assume the end effector is a dummy link (non active)
        links = self.links + [self.robot_ee]
        # if index is empty, return all
        if index is None:
            poses = [link.getTransform() for link in links]
        else:
            poses = [links[i].getTransform() for i in index]

        positions = [pose[1] for pose in poses]
        rotations = [matrix_to_quat(pose[0]) for pose in poses]
        return np.array(positions), np.array(rotations)

    def solve_ik(
        self,
        point,
        init_config=None,
        max_iters=100,
        tolerance=1e-3,
        none_on_fail=True,
    ):
        """Solve the inverse kinematics problem w/o initial angles"""
        # Randomly sample an initial configuration if not given
        if init_config is None:
            init_config = self.sample()
        config = np.array(self.robot.getConfig())
        config[self.active_joints] = init_config
        self.robot.setConfig(list(config))

        # Setup objective
        # TODO test this for rotation SO(3)
        # variable
        if self.rotation == "variable":
            obj = ik.objective(
                self.robot_ee,
                # target rotation matrix
                R=list(quat_to_matrix(point[3:7]).flatten()),
                t=list(point[:3]),
            )

        # free
        elif self.rotation == "free":
            obj = ik.objective(
                self.robot_ee,
                local=[0, 0, 0],
                world=list(point[:3]),
            )

        # fixed
        else:
            obj = ik.objective(
                self.robot_ee,
                # target rotation matrix
                R=list(self.fixed_rotation.flatten()),
                t=list(point[:3]),
            )

        # Solve
        solver = ik.solver(obj)
        solver.setMaxIters(max_iters)
        solver.setTolerance(tolerance)
        success = solver.solve()

        if success or not none_on_fail:
            # exclude the ee
            config = np.array(self.robot.getConfig())[self.active_joints]
            for i in self.cyclic_joints:
                config[i] = wrap_to_pi(config[i])
            return config
        else:
            return None


class KinematicChain(Robot):
    """A kinematic chain robot"""

    # Nothing needs to be overridden here


class Kinova(Robot):
    """Kinova Robotic Arm"""

    def __init__(self, name, domain, rot_domain, fixed_rotation=None):
        """Initialize the kinova robot. Mainly specify the active joints."""
        super().__init__(name, domain, rot_domain, fixed_rotation)

        # The active joints are the [1, 2, 3, 4, 5, 6, 7] joints
        active_joints = [1, 2, 3, 4, 5, 6, 7]
        self.init_attributes(active_joints)
        self.robot_ee = self.robot.link("Tool_Frame")

        # Get robot geometry for collision detection
        self.self_geometry = [
            self.robot.link(0).geometry(),
            self.robot.link(1).geometry(),
            self.robot.link(2).geometry(),
        ]
        self.ee_geometry = [
            self.robot.link("gripper:Link_0").geometry(),
            self.robot.link("gripper:Link_1").geometry(),
            self.robot.link("gripper:Link_2").geometry(),
            self.robot.link("gripper:Link_3").geometry(),
            self.robot.link("gripper:Link_4").geometry(),
            self.robot.link("gripper:Link_5").geometry(),
            self.robot.link("gripper:Link_6").geometry(),
            self.robot.link("gripper:Link_7").geometry(),
            self.robot.link("gripper:Link_8").geometry(),
        ]

    def solve_ik(
        self,
        point,
        init_config=None,
        max_iters=100,
        tolerance=1e-3,
        none_on_fail=True,
    ):
        """Solve the inverse kinematics problem w/o initial angles

        Consider self collision with the end effector
        """
        q = super().solve_ik(
            point, init_config, max_iters, tolerance, none_on_fail
        )

        # Check for self collision
        # still return none if in collision
        if q is not None and self.check_self_collision(q):
            return None

        return q

    def check_self_collision(self, q):
        """Check for self collision"""
        # return False
        # Set config
        config = np.array(self.robot.getConfig())
        config[self.active_joints] = q
        self.robot.setConfig(list(config))

        collision = collide.group_collision_iter(
            self.self_geometry, self.ee_geometry
        )
        return sum(1 for _ in collision) > 0

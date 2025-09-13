"""Module providing a class that defines a robot"""

import numpy as np

from ..grr.utils import se3_distance
from ..grr.utils import wrap_to_pi, interpolate_angle
from ..grr.utils import sample_quat, interpolate_quat
from ..grr.utils import euler_to_quat, quat_to_euler

import pybullet as p


# TODO
# Fix this


class Robot:
    """A meta robot class"""

    def __init__(self, robot_urdf, domain, rot_domain, fixed_rotation=None):
        """Initialize a robot in PyBullet instead
        given a domain and a rotation type

        Args:
            robot_urdf: the urdf path of the robot
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
        self.client = p.connect(p.DIRECT)
        self.robot = p.loadURDF(robot_urdf, physicsClientId=self.client)
        self.disable_all_collisions()
        num_joints = p.getNumJoints(self.robot, self.client)
        # Get the joint limits
        joint_limits = [self.get_joint_limit(i) for i in range(num_joints)]
        self.non_fixed_joints = self.get_active_joints(np.array(joint_limits))

        self.domain = domain
        self.rot_domain = rot_domain
        self.fixed_rotation = fixed_rotation
        # Fixed rotation provided, not totally free
        if self.fixed_rotation is not None:
            # convert euler to quaternion
            self.fixed_rotation = euler_to_quat(np.array(fixed_rotation))
            # rotation totally fixed
            if np.sum(rot_domain) == 0:
                self.rotation = "fixed"
            # some rotation is not fixed
            else:
                self.rotation = "variable"
        # Free rotation
        else:
            self.rotation = "free"

        # Get the active joints. By default these are the non-fixed joint.
        # But some robot does not necessarily take all the non-fixed joints
        # (For example, gripper joints could be ignored)
        active_joints = self.non_fixed_joints.copy()
        self.init_attributes(active_joints)
        # Assume the last joint of the robot
        self.robot_ee = num_joints - 1

    def disable_all_collisions(self):
        """Disable all collisions"""
        num_joints = p.getNumJoints(self.robot, self.client)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    continue
                p.setCollisionFilterPair(
                    self.robot, self.robot, i, j, 0, self.client
                )

    def init_attributes(self, active_joints):
        """Initialize the attributes"""
        self.active_joints = active_joints
        self.joint_limits = np.array(
            [self.get_joint_limit(i) for i in self.active_joints]
        )
        self.num_joints = len(self.active_joints)
        self.cyclic_joints = self.get_cyclic_joints(self.joint_limits)

        # This is mainly for the PyBullet IK solver
        # which always returns the joint values of all non-fixed joints.
        # The active joint indices w.r.t. all the non-fixed joints
        self.active_joint_indices = self.get_joint_indices_in_non_fixed(
            self.active_joints
        )
        # Get only the active ones
        # In pybullet, they share the same id
        self.links = self.active_joints.copy()

    def get_active_joints(self, limits):
        """Return the active joint (non-fixed joints)"""
        joints = []
        for i, limit in enumerate(limits):
            if limit[0] < limit[1]:
                joints.append(i)
        return joints

    def get_cyclic_joints(self, joint_limits):
        """Return the cyclic joint indices"""
        return np.array(
            [
                i
                for i, limit in enumerate(joint_limits)
                if (
                    (limit[0] == -np.inf and limit[1] == np.inf)
                    or (limit[0] == -1000 and limit[1] == 1000)
                )
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
        # variable
        if self.rotation == "variable":
            # if only need to sample in one rotation dimension
            # keep the other two as the same as fixed_rotation
            if np.sum(self.rot_domain) == 1:
                angle = np.random.uniform(-np.pi, np.pi)
                euler = quat_to_euler(self.fixed_rotation)
                index = self.rot_domain.index(1)
                euler[index] = angle
                quat = euler_to_quat(euler)

            # TODO this implementation is not totally correct
            elif np.sum(self.rot_domain) == 2:
                raise NotImplementedError
                # angle1 = np.random.uniform(-np.pi, np.pi)
                # angle2 = np.random.uniform(-np.pi, np.pi)
                # euler = quat_to_euler(self.fixed_rotation)
                # indices = [i for i, _ in enumerate(self.rot_domain) if x == 1]
                # euler[indices[0]] = angle1
                # euler[indices[1]] = angle2
                # quat = euler_to_quat(euler)

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
        for i, joint in enumerate(self.active_joints):
            self.set_joint(joint, config[i])

        # Acquire positions and rotations
        # assume the end effector is a dummy link (non active)
        links = self.links + [self.robot_ee]
        # if index is empty, return all
        if index is None:
            poses = [self.get_joint_pose(link) for link in links]
        else:
            poses = [self.get_joint_pose(links[i]) for i in index]

        positions = np.array([pose[0] for pose in poses])
        rotations = np.array([pose[1] for pose in poses])
        return positions, rotations

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
        # Set config
        for i, joint in enumerate(self.active_joints):
            self.set_joint(joint, init_config[i])

        # Setup objective
        # variable
        if self.rotation == "variable":
            result = self.ik(
                point[:3], point[3:7], max_iters=max_iters, tolerance=tolerance
            )

        # free
        elif self.rotation == "free":
            result = self.ik(
                point[:3], max_iters=max_iters, tolerance=tolerance
            )

        # fixed
        else:
            result = self.ik(
                point[:3],
                self.fixed_rotation,
                max_iters=max_iters,
                tolerance=tolerance,
            )

        # Check convergence
        for i, joint in enumerate(self.active_joints):
            self.set_joint(joint, result[i])
        # get current pose
        pos, rot = self.solve_fk(result, [-1])
        if self.rotation == "free":
            curr_point = pos[0]
            success = np.allclose(point, curr_point, atol=tolerance)
        else:
            curr_point1 = np.concatenate([pos[0], rot[0]])
            curr_point2 = np.concatenate([pos[0], -rot[0]])
            success = np.allclose(
                point, curr_point1, atol=tolerance
            ) or np.allclose(point, curr_point2, atol=tolerance)

        # Return result
        if success or not none_on_fail:
            for i in self.cyclic_joints:
                result[i] = wrap_to_pi(result[i])
            return result
        else:
            return None

    # PyBullet wrappers
    def get_joint_limit(self, joint):
        """Get the joint limit"""
        return p.getJointInfo(self.robot, joint, self.client)[8:10]

    def get_joint_indices_in_non_fixed(self, active_joints):
        """Get the joint indices in the non-fixed joints"""
        indices = []
        for i in active_joints:
            try:
                index = self.non_fixed_joints.index(i)
                indices.append(index)
            except ValueError as e:
                print(f"Active joint {i} is not in the non-fixed joints")
                raise e
        return indices

    def set_joint(self, joint, value):
        """Set the joint value"""
        p.resetJointStateMultiDof(
            self.robot, joint, [value], physicsClientId=self.client
        )

    def get_joint_position(self, joint):
        """Get the joint position"""
        return p.getJointState(self.robot, joint, self.client)[0]

    def get_joint_pose(self, joint):
        """Get the joint pose"""
        return p.getLinkState(self.robot, joint, 0, 0, self.client)[4:6]

    def get_link_from_name(self, link_name):
        """Get the link index from the link name"""
        num_joints = p.getNumJoints(self.robot, self.client)
        names = [
            p.getJointInfo(self.robot, i, self.client)[12]
            for i in range(num_joints)
        ]
        try:
            index = names.index(link_name.encode())
        except ValueError as e:
            print(f"Link {link_name} is not in the robot")
            raise e
        return index

    def ik(self, position, rotation=None, max_iters=100, tolerance=1e-3):
        """Solve the inverse kinematics problem"""
        # Get limits
        # ul = None
        # ll = None
        # Get current configuration
        # curr_config = [
        #     self.get_joint_position(joint) for joint in self.active_joints
        # ]
        if rotation is None:
            result = p.calculateInverseKinematics(
                self.robot,
                self.robot_ee,
                position,
                # lowerLimits=ll,
                # upperLimits=ul,
                # jointRanges=(ul - ll) + 1,
                # restPoses=curr_config,
                maxNumIterations=max_iters,
                residualThreshold=tolerance,
                physicsClientId=self.client,
            )
        else:
            result = p.calculateInverseKinematics(
                self.robot,
                self.robot_ee,
                position,
                rotation,
                # lowerLimits=ll,
                # upperLimits=ul,
                # jointRanges=(ul - ll) + 1,
                # restPoses=curr_config,
                maxNumIterations=max_iters,
                residualThreshold=tolerance,
                physicsClientId=self.client,
            )

        # Extract result
        # The IK solver function will return the joint values
        # of all non-fixed joints, which is not necessarily the same
        # as active joints defined in the robot class
        return np.array(result)[self.active_joint_indices]


class KinematicChain(Robot):
    """A kinematic chain robot"""

    # Nothing needs to be overridden here


class Kinova(Robot):
    """Kinova Robotic Arm"""


# # TODO the self_links and gripper_links are not correct
#     def __init__(self, robot_urdf, domain, rot_domain, fixed_rotation=None):
#         """Initialize the kinova robot. Mainly specify the active joints."""
#         super().__init__(robot_urdf, domain, rot_domain, fixed_rotation)

#         # The active joints are the [0, 1, 2, 3, 4, 5, 6] joints
#         active_joints = [0, 1, 2, 3, 4, 5, 6]
#         self.init_attributes(active_joints)
#         self.robot_ee = self.get_link_from_name("Tool_Frame")

#         # Get robot links for collision detection
#         # enabled specific collisions
#         self.self_links = [
#             self.robot.link(0).geometry(),
#             self.robot.link(1).geometry(),
#             self.robot.link(2).geometry(),
#         ]
#         self.gripper_links = [
#             self.robot.link("gripper:Link_0").geometry(),
#             self.robot.link("gripper:Link_1").geometry(),
#             self.robot.link("gripper:Link_2").geometry(),
#             self.robot.link("gripper:Link_3").geometry(),
#             self.robot.link("gripper:Link_4").geometry(),
#             self.robot.link("gripper:Link_5").geometry(),
#             self.robot.link("gripper:Link_6").geometry(),
#             self.robot.link("gripper:Link_7").geometry(),
#             self.robot.link("gripper:Link_8").geometry(),
#         ]
#         for i in self.self_links:
#             for j in self.gripper_links:
#                 p.setCollisionFilterPair(
#                     self.robot, self.robot, i, j, 1, self.client
#                 )

#     def solve_ik(
#         self,
#         point,
#         init_config=None,
#         max_iters=100,
#         tolerance=1e-3,
#         none_on_fail=True,
#     ):
#         """Solve the inverse kinematics problem w/o initial angles

#         Consider self collision with the end effector
#         """
#         q = super().solve_ik(
#             point, init_config, max_iters, tolerance, none_on_fail
#         )

#         # Check for self collision
#         # still return none if in collision
#         if q is not None:
#             # set joint
#             for i, joint in enumerate(self.active_joints):
#                 self.set_joint(joint, q[i])
#             # check collision
#             p.performCollisionDetection(self.client)
#             dists = list(
#                 point[8]
#                 for point in p.getContactPoints(physicsClientId=self.client)
#             )
#             if bool(dists) and np.min(dists) < 0:
#                 return None

#         return q


class UR10(Robot):
    """UR10 Robotic Arm with Robotis hand and D435 camera"""

    def __init__(self, robot_urdf, domain, rot_domain, fixed_rotation=None):
        """Initialize the ur10 robot. Mainly specify the active joints."""
        super().__init__(robot_urdf, domain, rot_domain, fixed_rotation)

        # The active joints are the [1, 2, 3, 4, 5, 6] joints
        active_joints = [1, 2, 3, 4, 5, 6]
        self.init_attributes(active_joints)
        self.robot_ee = self.get_link_from_name("ee_link")

        # Get robot links for collision detection
        # enabled specific collisions
        self.self_links = [1, 2, 3, 4, 5, 6]
        self.gripper_links = [8, 9, 10, 11, 12, 15]

        for i in self.self_links:
            for j in self.gripper_links:
                p.setCollisionFilterPair(
                    self.robot, self.robot, i, j, 1, self.client
                )

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
        if q is None:
            return q

        # Check for self collision
        # still return none if in collision
        # set joint
        for i, joint in enumerate(self.active_joints):
            self.set_joint(joint, q[i])
        # check collision
        p.performCollisionDetection(self.client)
        dists = list(
            point[8]
            for point in p.getContactPoints(physicsClientId=self.client)
        )
        if bool(dists) and np.min(dists) < 0:
            return None

        return q

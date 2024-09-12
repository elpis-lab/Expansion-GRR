"""Module providing a class that defines the global redundancy resolution"""

import pickle
import numpy as np
import networkx as nx

from .workspace import RedundancyWorkspace
from .solver import RedundancySolver
from .utils import wrap_to_pi


class RedundancyResolution:
    """A class to define the global redundancy resolution

    There are three main graphs used:
        - workspace graph: a graph that defines the workspace points.
        - solver graph: a graph that is used by the solver
        - resolution graph: a graph that contains the resolution
    The first two graphs contain the same nodes. Each node is a workspace point.

    Solver graph and resolution graphs both originate from the workspace graph.
    But the contents of the nodes are different:
        - solver graph: each node contains a list of configurations that
                        match the workspace point.
        - resolution graph: each node contains one single configuration
                            assigned at that workspace point.

    Also, their edges are different.
        - solver graph: an edge between two nodes means that the two nodes
                        are connected in the workspace.
        - resolution graph: an edge between two nodes means that the two nodes
                            are connected in the C space in global resolution.
    """

    def __init__(self, robot):
        """Initialize the redundancy resolution

        Args:
            robot: robot to perform redundancy resolution on
            sampling_method: method to use for workspace sampling
        """
        self.robot = robot

        self.graph = nx.Graph()
        self.nn = None

        self.workspace = RedundancyWorkspace(robot)
        self.solver = RedundancySolver(self.workspace, robot)

        # For teleoperation usage
        self.planning_mode = False
        self.plan_path = None
        self.path_index = 0

    def save_workspace_graph(self, graph_path, nn_path):
        """Save a workspace graph and nn search structure to a pickle file"""
        self.workspace.save_workspace_graph(graph_path, nn_path)

    def load_workspace_graph(self, graph_path, nn_path):
        """Load a workspace graph and nn search structure from a pickle file"""
        self.workspace.load_workspace_graph(graph_path, nn_path)

    def sample_workspace(
        self, n_pos_points, n_rot_points, sampling_method="random"
    ):
        """Sample n_points workspace points in the robot workspace

        Args:
            n_points: number of points to sample
            sampling_method: method to use for sampling
        """
        self.workspace.sample_workspace(
            n_pos_points, n_rot_points, sampling_method
        )

    def visualize_workspace_graph(self):
        """Visualize the workspace graph"""
        self.workspace.visualize_workspace_graph()

    def save_solver_graph(self, path):
        """Save a solver graph to a pickle file"""
        self.solver.save_solver_graph(path)

    def load_solver_graph(self, path):
        """Load a solver graph from a pickle file"""
        self.solver.load_solver_graph(path)

    # Random-GRR method
    # def sample_configuration_space(self, n_configs):
    #     """Sample n_configs configurations in the robot configuration space

    #     Args:
    #         n_configs: number of configs to sample at each workspace point
    #     """
    #     self.solver.init_graph(self.workspace.graph)
    #     self.solver.sample_configuration_space(n_configs)

    # def solve_resolution(self, n_random_descents=2, n_initial_guess=10):
    #     """Solve the CSP to find asssignments as resolution"""
    #     self.solver.solve_resolution(n_random_descents, n_initial_guess)

    def global_expansion(self, configs):
        """Sample by expanding from a set of configurations

        Args:
            configs: a list of configurations to expand from
        """
        self.solver.init_graph(self.workspace.graph)
        self.solver.global_expansion(configs)

    # TODO Future Improvement
    def fix_boundary(self, n_neighbor_layer=1, n_iter=5):
        """Fix the discontinuous boundary of the resolution graph"""
        self.solver.fix_boundary(n_neighbor_layer, n_iter)

    # def optimize_resolution(self, n_length_optimization=5):
    #     """Optimize the current resolution"""
    #     self.solver.optimize_resolution(n_length_optimization)

    def build_resolution_graph_and_nn(self, build_new_nn=True):
        """Build the resolution graph from the solver graph"""
        # Get the resolution graph from the solver
        res = self.solver.build_resolution_graph_and_nn(build_new_nn)
        if build_new_nn:
            self.graph = res[0]
            self.nn = res[1]
        else:
            self.graph = res

    def save_resolution_graph(self, graph_path, nn_path):
        """Save the graph to a file"""
        pickle.dump(self.graph, open(graph_path, "wb"))
        pickle.dump(self.nn, open(nn_path, "wb"))

    def load_resolution_graph(self, graph_path, nn_path):
        """Load the graph from a file"""
        self.graph = pickle.load(open(graph_path, "rb"))
        self.nn = pickle.load(open(nn_path, "rb"))

        # Logging
        print("\nResolution graph loaded")
        print("Graph has", self.graph.number_of_nodes(), "nodes")
        print("Graph has", self.graph.number_of_edges(), "edges")

    def teleop_solve(self, target_point, curr_config, max_change=0.03):
        """Teleoperation solve"""
        # Get the current point and config
        pos, rot = self.robot.solve_fk(curr_config, [-1])
        curr_point = pos[0]
        if self.robot.rotation == "variable":
            curr_point = np.concatenate([pos[0], rot[0]])

        # Projection from neighbors
        q = self.solve(target_point, curr_config, none_on_fail=True)

        # If not initialized yet
        if curr_config is None:
            return q

        # A new valid target is found
        if q is not None:
            # Check if from current to new q is valid
            if self.solver.is_continuous(
                self.robot, curr_config, q, curr_point, target_point
            ):
                self.plan_path = None
                self.path_index = 0
                return self.teleop_towards(curr_config, q, max_change)

            else:
                # Need to plan a path towards q
                if self.plan_path is None:
                    self.plan_path, _ = self.plan(
                        curr_point, target_point, interpolation=1
                    )
                    if len(self.plan_path) == 0:
                        return curr_config
                    else:
                        self.path_index = 1
                        return self.teleop_towards(
                            curr_config, self.plan_path[1], max_change
                        )
                # Path is already planned, follow it
                else:
                    self.path_index += 1
                    if self.path_index < len(self.plan_path):
                        return self.teleop_towards(
                            curr_config,
                            self.plan_path[self.path_index],
                            max_change,
                        )
                    else:
                        self.plan_path = None
                        self.path_index = 0
                        return curr_config

        # Self-collision happens or Discontinuity encountered
        else:
            n_neighbors = 5
            # Try to go to the nearest neighbors on the roadmap grid
            neighbors = self.workspace.get_workspace_neighbors(
                target_point, self.nn, k=n_neighbors
            )
            for neighbor in neighbors:
                q = self.graph.nodes[neighbor]["config"]
                p = self.graph.nodes[neighbor]["point"]
                # check path validity
                if self.solver.is_continuous(
                    self.robot, q, curr_config, p, curr_point
                ):
                    return self.teleop_towards(curr_config, q, max_change)

        return None

    def teleop_towards(self, curr_config, target_config, max_change):
        """Move the curr_config towards the target_config given the max_change"""
        # Joint difference
        diff = target_config - curr_config
        # For cyclic joints, wrap the difference to [-pi, pi)
        for i in self.robot.cyclic_joints:
            diff[i] = wrap_to_pi(diff[i])
        # Check the difference
        diff = np.abs(diff)
        if np.max(diff) < max_change:
            return self.robot.interpolate(curr_config, target_config, 1)
        else:
            u = max_change / np.max(diff)
            return self.robot.interpolate(curr_config, target_config, u)

    # Similar to project_neighbors(),
    # but the way of collecting neighbors is different
    def solve(
        self,
        point,
        curr_config=None,
        nearest_node_only=False,
        regular_ik=False,
        none_on_fail=False,
    ):
        """Solve redundancy for a given point in the workspace
        Uses inverse distance weighted interpolation at the workspace nodes to
        derive a reasonable guess for the IK solution

        Args:
            point: point in the workspace
            curr_config: current configuration of the robot
            nearest_node_only: whether to only use the configuration of the
                               nearest graph node as the guess

        Returns:
            q: configuration of the robot
        """

        def solve_with_guess(guess):
            """A helper function to solve IK with a guess configuration"""
            return self.robot.solve_ik(point, guess, none_on_fail=none_on_fail)

        if regular_ik:
            return solve_with_guess(curr_config)

        neighbors = self.workspace.get_workspace_neighbors(
            point, self.nn, k=self.workspace.interpolate_num_neighbors
        )
        neighbors = [
            n for n in neighbors if self.graph.nodes[n]["config"] is not None
        ]

        # If no neighbor is found, simply use current configuration
        if len(neighbors) == 0:
            return solve_with_guess(curr_config)

        # If only get solutions from the nearest graph node
        if nearest_node_only:
            return self.graph.nodes[neighbors[0]]["config"]
            # return solve_with_guess(self.graph.nodes[neighbors[0]]["config"])

        # If it falls on a node, use that node's configuration
        if (
            self.robot.workspace_distance(
                point, self.graph.nodes[neighbors[0]]["point"]
            )
            < 1e-3
        ):
            return solve_with_guess(self.graph.nodes[neighbors[0]]["config"])

        # Get a subgraph of the neighbors
        subgraph = self.graph.subgraph(neighbors)
        # find the largest connected component with cloest neighbor
        ccs = list(nx.connected_components(subgraph))
        for cc in ccs:
            if neighbors[0] in cc:
                component = cc
                break

        q_neighbors = [self.graph.nodes[j]["config"] for j in component]
        p_neighbors = [self.graph.nodes[j]["point"] for j in component]
        distances = np.array(
            [self.robot.workspace_distance(point, p) for p in p_neighbors]
        )

        # Compute weights
        max_dist = np.max(distances)
        weights = [(max_dist / d) ** 2 for d in distances]

        # Compute and project the average configuration
        q_avg = self.robot.average(q_neighbors, weights)

        return solve_with_guess(q_avg)

    def plan(self, start_point, goal_point, interpolation=8):
        """Plan a trajectory from start_point to goal_point"""
        # Get the nearest nodes
        neighbors1 = self.workspace.get_workspace_neighbors(
            start_point, self.nn, k=1
        )
        neighbors2 = self.workspace.get_workspace_neighbors(
            goal_point, self.nn, k=1
        )
        # Select a node in graph for start and goal points
        # The node is valid as long as star or goal point can directly
        # interpolate to the node
        num_div = 8
        n1, n2 = None, None
        for neighbor in neighbors1:
            for k in range(num_div):
                sub_point = self.robot.workspace_interpolate(
                    start_point,
                    self.graph.nodes[neighbor]["point"],
                    k / num_div,
                )
                sub_waypoint = self.solve(sub_point)
                if sub_waypoint is None:
                    break
            else:
                n1 = neighbor
                break
        for neighbor in neighbors2:
            for k in range(num_div):
                sub_point = self.robot.workspace_interpolate(
                    goal_point,
                    self.graph.nodes[neighbor]["point"],
                    k / num_div,
                )
                sub_waypoint = self.solve(sub_point)
                if sub_waypoint is None:
                    break
            else:
                n2 = neighbor
                break

        # If no valid neighbor is found, do not plan
        if n1 is None or n2 is None:
            print("No valid neighbor found")
            return [], []

        # Get the shortest path in the graph
        try:
            path = nx.shortest_path(
                self.graph,
                source=n1,
                target=n2,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            print("No path found")
            return [], []

        # Add start_point and goal_point
        path_points = [self.graph.nodes[p]["point"] for p in path]
        path_points = [start_point] + path_points + [goal_point]

        # Interpolate along the path
        w_path = []
        c_path = []
        for point_i, point_j in zip(path_points[:-1], path_points[1:]):
            for k in range(interpolation):
                sub_point = self.robot.workspace_interpolate(
                    point_i,
                    point_j,
                    k / interpolation,
                )
                sub_waypoint = self.solve(sub_point)
                # should not happen
                if sub_waypoint is None:
                    continue
                w_path.append(sub_point)
                c_path.append(sub_waypoint)
        # interpolation doesn't consider the last point
        w_path.append(goal_point)
        c_path.append(self.solve(goal_point))

        return np.array(c_path), np.array(w_path)

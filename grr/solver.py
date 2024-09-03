"""Module providing a class that defines the global redundancy solver"""

import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import deque


class RedundancySolver:
    """Perform the global redundancy solving procedure

    This class mainly is used to run the proposed algorithm to solve
    the redundancy resolution problem.
    """

    def __init__(self, workspace, robot):
        """
        Args:
            resolution: the redundancy resolution object

        Attributes:
            graph: A graph (nx.Graph()) with node attributes:
                - "point": the workspace point parameters
                - "q_list": a list of potential associated configurations
                - "config": the configuration assigned to the node
                and edge attributes:
                - "connected": whether the edge is connected in the solution
        """
        self.workspace = workspace
        self.robot = robot
        self.graph = None

    def save_solver_graph(self, path):
        """Save the graph to a file"""
        pickle.dump(self.graph, open(path, "wb"))

    def load_solver_graph(self, path):
        """Load a graph from a file"""
        self.graph = pickle.load(open(path, "rb"))

        # Logging
        print("\nSolver graph loaded")
        print("Graph has", self.graph.number_of_nodes(), "nodes")
        n_configs = 0
        n_connected = 0
        for _, node in self.graph.nodes(data=True):
            n_configs += len(node["q_list"])
        for _, _, edge in self.graph.edges(data=True):
            n_connected += len(edge["q_list"])
        print("Graph has", n_configs, "configurations")
        print(n_connected, "pairs of configurataion are connected")

    def init_graph(self, workspace_graph):
        """Initialize the graph from workspace graph

        Args:
            clear: clear the previous resolution or not
        """
        # Copy from workspace graph
        self.graph = nx.Graph()
        for i, node in workspace_graph.nodes(data=True):
            self.graph.add_node(i, point=node["point"], q_list=[], config=None)
        for i, j, edge in workspace_graph.edges(data=True):
            self.graph.add_edge(
                i, j, weight=edge["weight"], q_list=set(), connected=False
            )

    def global_expansion(self, configs):
        """Sample by expanding from a set of configurations

        Args:
            configs: a list of configurations to expand from
        """
        print(
            "\nSampling configuration for each point by expanding from"
            + " a set of starting configurations"
        )

        # TODO Future Improvement
        # Load initial configurations from existing graph
        # graph = pickle.load(
        #     open("graph/kinova/rot_free/graph_resolution.pickle", "rb")
        # )
        # configs = [
        #     node["config"]
        #     for i, node in graph.nodes(data=True)
        #     if node["config"] is not None
        # ]

        # Initialize the graph
        start_neighbors = self.initialize_from_configs(configs)
        if len(start_neighbors) == 0:
            print("No valid start configurations")
            return

        # Repeat expansion until no more nodes can be updated
        while True:
            continue_expansion = False

            # Run BFS to expand
            queue = deque(start_neighbors)
            distance = {node: 1 for node in start_neighbors}
            next_distance = 2

            # # k neighbors
            # k = self.workspace.interpolate_num_neighbors * 4
            # k layers of neighbors
            k = 4

            pbar = tqdm(total=self.graph.number_of_nodes())
            while queue:
                pbar.update(1)
                i = queue.popleft()

                # Result worse, why?
                # # Shuffle if we already explore all nodes in this level
                # # and going to the next level
                # if distance[i] == next_distance:
                #     next_distance += 1
                #     np.random.shuffle(queue)

                # Add neighbors to queue
                for j in self.graph.neighbors(i):
                    if j in distance:
                        continue
                    queue.append(j)
                    distance[j] = distance[i] + 1

                # Expand this node
                config = self.graph.nodes[i]["config"]
                if config is not None:
                    continue

                q = self.project_neighbors(i, k)
                if q is None:
                    continue

                # Assign the configuration to this node
                self.graph.nodes[i]["config"] = q
                self.graph.nodes[i]["q_list"] = [q]
                continue_expansion = True

                # Check connectivity
                self.check_neighbor_connection(i)

            if not continue_expansion:
                break

        # Logging
        pbar.close()
        print("\nSolver graph built")
        n_configs = 0
        n_connected = 0
        for _, node in self.graph.nodes(data=True):
            if node["config"] is not None:
                n_configs += 1
        for _, _, edge in self.graph.edges(data=True):
            if edge["connected"]:
                n_connected += 1
        print("Sampled", n_configs, "configurations successfully")
        print(n_connected, "pairs of configurataion are connected")

    def initialize_from_configs(self, configs):
        """Initialize the starting nodes with the starting configurations"""
        start_neighbors = set()
        valid_count = 0
        for _, config in enumerate(configs):
            # Get the corresponding workspace point
            positions, rotations = self.robot.solve_fk(config, [-1])
            point = positions[0]
            if self.robot.rotation == "variable":
                point = np.concatenate([positions[0], rotations[0]])

            # Get the closest start point in the workspace graph
            start_node = self.workspace.get_workspace_neighbors(
                point, self.workspace.nn, k=1
            )[0]

            # Acquire candidates
            start_point = self.graph.nodes[start_node]["point"]
            start_config = self.robot.solve_ik(start_point, config)

            # needs to be valid
            if start_config is None:
                print("Cannot start with this configuration:", config)
                continue
            # should not deviate too much from the given configuration
            # if (
            #     self.robot.distance(config, start_config)
            #     > 0.1 * self.robot.num_joints
            # ):
            #     print(
            #         "This configuration deviates too much from the desired "
            #         + "starting configuration:",
            #         config,
            #     )
            #     continue

            # Assign config to the start node
            self.graph.nodes[start_node]["config"] = start_config
            self.graph.nodes[start_node]["q_list"] = [start_config]
            # Check connectivity
            self.check_neighbor_connection(start_node)

            valid_count = valid_count + 1

            # Add neighbors to start neighbors
            for node in self.graph.neighbors(start_node):
                start_neighbors.add(node)

        print(
            "Valid start configurtioons:",
            valid_count,
            100 * valid_count / len(configs),
            "%",
        )

        return start_neighbors

    def project_neighbors(self, i, k):
        """Project the weighted average configurations of neighbors
        to the desired workspace point
        """
        point = self.graph.nodes[i]["point"]

        # Get vaild neighbors
        neighbors = [
            j
            for j in self.find_k_layers_neighbors(i, k)
            # for j in self.workspace.get_workspace_neighbors(
            #     point, self.workspace.nn, k=k
            # )
            if i != j and self.graph.nodes[j]["config"] is not None
        ]
        if len(neighbors) == 0:
            return None

        # Collect configurations and points of neighbors
        q_neighbors = [self.graph.nodes[j]["config"] for j in neighbors]
        p_neighbors = [self.graph.nodes[j]["point"] for j in neighbors]
        distances = np.array(
            [self.robot.workspace_distance(point, p) for p in p_neighbors]
        )

        # Compute weights
        max_dist = np.max(distances)
        weights = [(max_dist / d) ** 2 for d in distances]

        # Compute and project the average configuration
        q_avg = self.robot.average(q_neighbors, weights)
        q = self.robot.solve_ik(point, q_avg)
        return q

    def find_k_layers_neighbors(self, i, k):
        """Find the k layers neighbors of a node i in the graph"""
        # Repeat for k layers
        visited = set([i])
        current_layer = set([i])
        for _ in range(k):

            # Prepare to collect the next layer of neighbors
            next_layer = set()
            for node in current_layer:
                # Get neighbors of the current node
                neighbors = set(self.graph.neighbors(node))
                # Add neighbors to the next layer, excluding visited nodes
                next_layer.update(neighbors - visited)

            # Add the next layer to visited nodes
            visited.update(next_layer)
            # Move to the next layer
            current_layer = next_layer

        visited.remove(i)
        return visited

    def check_neighbor_connection(self, i):
        """Test and connect the neighbors of a node i"""
        if self.graph.nodes[i]["config"] is None:
            return

        for j in self.graph.neighbors(i):
            if self.graph.nodes[j]["config"] is None:
                continue

            continuous = self.is_continuous(
                self.robot,
                self.graph.nodes[i]["config"],
                self.graph.nodes[j]["config"],
                self.graph.nodes[i]["point"],
                self.graph.nodes[j]["point"],
            )
            self.graph.edges[i, j]["connected"] = continuous
            if continuous:
                self.graph.edges[i, j]["q_list"] = [(0, 0)]

    @staticmethod
    def is_continuous(robot, q1, q2, point1, point2):
        """Check if two configurations follow the continuous constraints

        Make this a static method so that it can be used in multiprocessing
        """
        scale = np.sqrt(robot.num_joints)
        # return RedundancySolver.is_continuous_bisect(
        #     robot, q1, q2, point1, point2, scale, scale * 5e-2
        # )
        return RedundancySolver.is_continuous_bisect(
            robot, q1, q2, point1, point2, 1.8, scale * 5e-2
        )

    @staticmethod
    def is_continuous_bisect(
        robot, q1, q2, point1, point2, deviation, epsilon=5e-2
    ):
        """Check if two configurations follow the continuous constraints

        Linearly interpolate and bisectionally visit and check
        with the help of a queue to avoid stack overflow issues
        """
        n_divs = int(np.ceil(robot.distance(q1, q2) / epsilon))
        queue = deque()
        queue.append((q1, q2, 0, n_divs + 1))

        while len(queue) > 0:
            qa, qb, ia, ib = queue.popleft()
            d = robot.distance(qa, qb)

            # If the two points are already adjacent
            if ib == ia + 1:
                continue

            # Find the middle point and configuration
            im = (ia + ib) // 2
            qm = robot.solve_ik(
                robot.workspace_interpolate(point1, point2, im / (n_divs + 1)),
                robot.interpolate(qa, qb, (im - ia) / (ib - ia)),
                none_on_fail=False,  # Only consider None due to collision
            )
            if qm is None:
                return False

            # Check middle's deviation from "straight path"
            d1 = robot.distance(qa, qm)
            if d1 > deviation * d:
                return False
            d2 = robot.distance(qm, qb)
            if d2 > deviation * d:
                return False

            # Add to queue for further bisection
            queue.append((qa, qm, ia, im))
            queue.append((qm, qb, im, ib))
        return True

    def clear_resolution(self):
        """Reinitialize resolution but keep the workspace graph"""
        # Clear resolution stored in nodes and edges
        for _, node in self.graph.nodes(data=True):
            node["config"] = None
        for _, _, edge in self.graph.edges(data=True):
            edge["connected"] = False

    def build_resolution_graph_and_nn(self, build_new_nn=True):
        """Make a resolution graph and nearest neighbor structure
        from the current result
        """
        print("Building resolution graph from solver graph")
        graph = nx.Graph()
        new_id = {}
        for i, node in self.graph.nodes(data=True):
            if node["config"] is not None:
                n = graph.number_of_nodes()
                new_id[i] = n
                graph.add_node(
                    n,
                    point=node["point"],
                    config=node["config"],
                )
        for i, j, edge in self.graph.edges(data=True):
            if edge["connected"]:
                graph.add_edge(new_id[i], new_id[j], weight=edge["weight"])

        if build_new_nn:
            print("Building nearest neighbor structure for new resolution")
            nn = self.workspace.build_nn(graph)
            return graph, nn
        else:
            return graph

    # TODO Future improvement
    def fix_boundary(self, n_neighbor_layer=1, n_iter=5):
        """Fix discontinunous boundary with destruction and rebuilding"""
        print("\nFixing discontinous boundary with destruct and rebuild")

        for _ in tqdm(range(n_iter)):
            # Find the discontinuous boundary
            disconnected_edges = set()
            boundary_nodes = set()
            for i, j, edge in self.graph.edges(data=True):
                if (
                    not edge["connected"]
                    and len(self.graph.nodes[i]["q_list"]) != 0
                    and len(self.graph.nodes[j]["q_list"]) != 0
                ):
                    disconnected_edges.add((i, j))
                    boundary_nodes.add(i)
                    boundary_nodes.add(j)
            if len(disconnected_edges) == 0:
                tqdm.write("No discontinuous nodes anymore")
                return
            else:
                tqdm.write("Discontinuous nodes: " + str(len(boundary_nodes)))

            # BFS to find the nodes to be destructed
            queue = deque(list(boundary_nodes))
            distances = {node: 0 for node in boundary_nodes}
            node_all_level = []
            while queue and len(node_all_level) < n_neighbor_layer:
                node_at_level = []
                for _ in range(len(queue)):
                    i = queue.popleft()
                    node_at_level.append(i)

                    for j in self.graph.neighbors(i):
                        if (
                            j in distances
                            or len(self.graph.nodes[j]["q_list"]) == 0
                        ):
                            continue
                        queue.append(j)
                        distances[j] = distances[i] + 1

                node_all_level.append(node_at_level)

            # Destruct nodes
            old_config = {}
            for nodes in node_all_level:
                for i in nodes:
                    for j in self.graph.neighbors(i):
                        self.graph.edges[i, j]["connected"] = False
                    old_config[i] = self.graph.nodes[i]["config"]
                    self.graph.nodes[i]["config"] = None

            # Rebuild - starting from outer loop
            # # k neighbors
            # k = (
            #     self.workspace.interpolate_num_neighbors
            #     * n_neighbor_layer
            #     // 4
            # )
            # k layers of neighbors
            k = 4

            # Count
            count = 0
            for _, nodes in enumerate(node_all_level):
                for i in nodes:
                    count += 1
            pbar = tqdm(total=count)

            node_all_level = node_all_level[::-1]
            for _, nodes in enumerate(node_all_level):
                for i in nodes:
                    pbar.update(1)

                    # Current goal
                    q_goal = self.project_neighbors(i, k)
                    if q_goal is None:
                        continue

                    # Check connectivity
                    self.graph.nodes[i]["config"] = q_goal
                    self.check_neighbor_connection(i)
            pbar.close()

            # For those nodes that not assigned a configuration
            for _, nodes in enumerate(node_all_level):
                for i in nodes:
                    if self.graph.nodes[i]["config"] is None:
                        self.graph.nodes[i]["config"] = old_config[i]
                        self.check_neighbor_connection(i)

            # self.optimize_length(1)

    # def optimize_resolution(self, n_length_optimization=5):
    #     """Optimize the resolution by optimizing the resolution length"""
    #     # Optimize the resolution length
    #     print("\nOptimizing resolution length")
    #     self.optimize_length(n_length_optimization)

    # def optimize_length(self, num_iters=20, ignore_disconnection=False):
    #     """Optimizes the configurations of the current resolution to minimize
    #     configuration space path lengths using coordinate descent.

    #     Args:
    #         num_iters: number of iteration to optimize the
    #     """

    #     def optimize_node(i):
    #         """Helper function to optimize a single node i.
    #         Return the total distance to the neighbors after optimization.
    #         """
    #         point = self.graph.nodes[i]["point"]
    #         config = self.graph.nodes[i]["config"]
    #         if config is None:
    #             return 0

    #         # Get neighbors
    #         if not ignore_disconnection:
    #             neighbors = [
    #                 j
    #                 for j in self.graph.neighbors(i)
    #                 if self.graph.edges[i, j]["connected"]
    #             ]
    #             original_connectivity = [True] * len(neighbors)
    #         else:
    #             neighbors = [
    #                 j
    #                 for j in self.graph.neighbors(i)
    #                 if self.graph.nodes[j]["config"] is not None
    #             ]
    #             original_connectivity = [
    #                 self.graph.edges[i, j]["connected"] for j in neighbors
    #             ]
    #         if len(neighbors) == 0:
    #             return 0

    #         # Get neighbors' configurations and points
    #         q_neighbors = [self.graph.nodes[j]["config"] for j in neighbors]
    #         p_neighbors = [self.graph.nodes[j]["point"] for j in neighbors]

    #         # Get current and average
    #         dist = np.sum(self.robot.distance(config, q) for q in q_neighbors)
    #         q_avg = self.robot.average(q_neighbors)

    #         # Try to move towards average
    #         max_tries = 10
    #         for _ in range(max_tries):
    #             # Current goal
    #             q_goal = self.robot.solve_ik(point, q_avg)
    #             if q_goal is None:
    #                 continue

    #             # Check if current goal is valid
    #             valid = True
    #             # check new distance
    #             d = np.sum(self.robot.distance(q_goal, q) for q in q_neighbors)
    #             if d > dist:
    #                 valid = False
    #             # check connectivity
    #             else:
    #                 connectivity = []
    #                 for q, p in zip(q_neighbors, p_neighbors):
    #                     connectivity.append(
    #                         self.is_continuous(
    #                             self.robot, q_goal, q, point, p)
    #                         )
    #                     )

    #                 if not ignore_disconnection:
    #                     if not all(connectivity):
    #                         valid = False
    #                 else:
    #                     if sum(connectivity) < len(original_connectivity):
    #                         valid = False
    #                     else:
    #                         for index, j in enumerate(neighbors):
    #                             self.graph.edges[i, j]["connected"] = (
    #                                 connectivity[index]
    #                             )

    #                 # Move to goal
    #                 if valid:
    #                     self.graph.nodes[i]["config"] = q_goal
    #                     return d

    #             # not moved, try subdividing
    #             q_avg = self.robot.interpolate(config, q_avg, 0.5)

    #         return dist

    #     # Optimize procedure
    #     # Optimize in random order
    #     nodes = list(self.graph.nodes())
    #     for _ in tqdm(range(num_iters)):
    #         sum_dist = 0

    #         # for i in tqdm(self.graph.nodes(), leave=False):
    #         #     sum_dist += optimize_node(i)
    #         np.random.shuffle(nodes)
    #         for i in tqdm(nodes):
    #             sum_dist += optimize_node(i)

    #         tqdm_str = "Changed average path length to " + str(
    #             round(sum_dist / self.graph.number_of_edges(), 4)
    #         )
    #         tqdm.write(tqdm_str)

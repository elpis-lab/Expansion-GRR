"""Module providing a class that defines the robot workspace graph"""

import pickle
import numpy as np
import networkx as nx
from sklearn.neighbors import BallTree
from pynndescent import NNDescent

from tqdm import tqdm
import matplotlib.pyplot as plt

from grr.utils import se3_metric
from grr.utils import get_staggered_grid, get_so3_grid
from grr.utils import quat_to_euler, rotvec_to_quat, quat_to_rotvec


class RedundancyWorkspace:
    """A class to define the global redundancy resolution workspace

    This class mainly is used to sample in workspace and connect the samples.
    The built graph will be used as the starting point for both graphs
    in the redundancy resolution.
    """

    def __init__(self, robot):
        """Initialize the redundancy resolution

        Args:
            robot: robot to perform redundancy resolution on

        Attributes:
            interpolate_num_neighbors: number of workspace neighbors to sample
                during interpolation. This is a recommended number.
            graph: A graph (nx.Graph()) with node attributes:
                - "point": the workspace point parameters
            nn: nearest neighbor search structure for searching in workspace
        """
        self.robot = robot
        self.pos_dims = len([1 for (a, b) in self.robot.domain if a != b])
        self.rot_dims = sum(self.robot.rot_domain)
        self.interpolate_num_neighbors = self.get_interpolate_num_neighbors()

        self.graph = nx.Graph()  # workspace graph
        self.nn = None  # nearest neighbor for searching

    def get_interpolate_num_neighbors(self):
        """Get the number of neighbors to sample during interpolation"""
        k = 2**self.pos_dims + self.rot_dims * 2
        return k

    def save_workspace_graph(self, graph_path, nn_path):
        """Save a workspace graph and nn search structure to a pickle file"""
        pickle.dump(self.graph, open(graph_path, "wb"))
        pickle.dump(self.nn, open(nn_path, "wb"))

    def load_workspace_graph(self, graph_path, nn_path):
        """Load a workspace graph and nn search structure from a pickle file"""
        self.graph = pickle.load(open(graph_path, "rb"))
        self.nn = pickle.load(open(nn_path, "rb"))

        # Logging
        print("\nWorkspace graph loaded")
        print("Graph has", self.graph.number_of_nodes(), "nodes")
        print("Graph has", self.graph.number_of_edges(), "edges")

    def build_nn(self, graph):
        """Build nearest neighbor search structure"""
        print("Building nearest neighbor search structure")
        # By default, when only dealing with position,
        # use BallTree for searching
        if self.robot.rotation != "variable":
            print("Building BallTree")
            nn = BallTree(
                np.array(
                    [node["point"] for _, node in graph.nodes(data=True)]
                ),
                metric="euclidean",
            )

        # When dealing with rotation, use pynndescent for searching
        # the trade-off is that pynndescent takes longer to build the tree,
        # but with customized SE3 distance, it is much faster to search
        else:
            print("Building NNDescent")
            print(
                "This structure would take a while to build,",
                "but it is much faster to search with SE(3) distance later.",
                "\nFor reference: 10K -> 40s, 100K -> 3mins, 1M -> 30mins",
            )
            nn = NNDescent(
                np.array(
                    [node["point"] for _, node in graph.nodes(data=True)]
                ),
                metric=se3_metric,
                n_neighbors=100,  # recommended number
            )

        return nn

    def sample_workspace(
        self, n_pos_points, n_rot_points, sampling_method="grid"
    ):
        """Sample n_points workspace points in the robot workspace

        Args:
            n_points: number of points to sample
            sampling_method: method to use for sampling
        """
        # Logging
        print("\nSample workspace in", self.pos_dims, "position dimension")
        print("and", self.rot_dims, "rotation dimensions")

        print("Sample with method:", sampling_method)
        print("Sample", n_pos_points, "position points")
        for i, (a, b) in enumerate(self.robot.domain):
            print("Dimension", i + 1, "range:", a, "->", b)
        print("Sample", n_rot_points, "rotation points")
        for i, a in enumerate(self.robot.rot_domain):
            to_sample = "Yes" if a else "No"
            print("Dimension", i + 1, "sample?", to_sample)

        # Clear
        self.graph.clear()

        # Sample in the workspace with random sampling
        # When using random sampling, each sampled point is itself a
        # combination of position and rotation, and the total
        # number of points is (n_pos_points * n_rot_points)
        if sampling_method == "random":
            n_total = n_pos_points * n_rot_points

            # Sample
            print("Sampling points from sampling")
            for i in tqdm(range(n_total)):
                point = self.robot.workspace_sample()
                self.add_workspace_node(point)

            # Build tree for searching
            self.nn = self.build_nn(self.graph)

            # Connect
            print("Connecting points")
            # automatically compute the appropriate number of neighbors
            constant = np.e / 4  # np.e
            k = int(
                constant * (1 + 1.0 / (self.pos_dims)) * np.log(n_pos_points)
            )
            if self.rot_dims > 0:
                k *= self.rot_dims * 2
            for i, node in tqdm(self.graph.nodes(data=True)):
                neighbors = self.get_workspace_neighbors(
                    node["point"], self.nn, k=k + 1
                )
                # Check validity of the neighbors and connect
                for j in neighbors:
                    if i != j:
                        self.add_workspace_edge(i, j)

        # Sample in the workspace with grid sampling
        # When using grid sampling, grid for position and rotation
        # are created separately and then combined by "multiplying" them.
        # The total number of points is (n_pos_points * n_rot_points)
        elif sampling_method == "grid":
            # Sample
            print("Sampling position points")
            # Get points and edges from a staggered grid
            # points are the workspace coordinates and edges are in index form
            points, edges = get_staggered_grid(n_pos_points, self.robot.domain)

            # If there is no need to sample in rotation,
            # directly add points to graph and connect position neighbors
            if self.robot.rotation != "variable" or n_rot_points <= 0:
                # Add nodes
                for point in tqdm(points):
                    self.add_workspace_node(point)

                # Build tree for searching
                self.nn = self.build_nn(self.graph)

                # Connect nodes
                print("Connecting position points")
                # add edges
                for i, j in tqdm(edges):
                    self.add_workspace_edge(i, j)

            # Else, need to sample rotation points
            else:
                # Get points by evenly distribute points on SO(3)
                print("Sampling rotation points")
                # automatically compute a good number of neighbors
                n_rot_neighbors = self.rot_dims * 2
                rotation_points, rotation_eges = get_so3_grid(
                    n_rot_points,
                    domain=self.robot.rot_domain,
                    num_neighbors=n_rot_neighbors,
                )

                # Add nodes
                # combine rotation and position points by Cartesian Product
                print("Combining position and rotation points")
                nodes = {}
                p_bar = tqdm(total=len(points) * len(rotation_points))
                for p, point in enumerate(points):
                    for r, rot_point in enumerate(rotation_points):
                        p_bar.update(1)
                        self.add_workspace_node(
                            np.concatenate((point, rot_point))
                        )
                        # store in nodes for later edge connection
                        nodes[(p, r)] = self.graph.number_of_nodes() - 1
                p_bar.close()

                # Build tree for searching
                self.nn = self.build_nn(self.graph)

                # Connect nodes
                # connect them with the rule of Cartesian Product
                print("Connecting points")
                # for a new combined point, the new edges are defined as
                # 1, position neighbors with the same rotation (4 / 6, 8)
                # 2, rotation neighbors with the same position (2 / 6)
                combined_edges = []
                for i, j in edges:
                    for r in range(n_rot_points):
                        combined_edges.append((nodes[(i, r)], nodes[(j, r)]))
                for i, j in rotation_eges:
                    for p in range(n_pos_points):
                        combined_edges.append((nodes[(p, i)], nodes[(p, j)]))

                # add edges
                for i, j in tqdm(combined_edges):
                    self.add_workspace_edge(i, j)

        else:
            raise ValueError("Unknown method:", sampling_method)

        # Logging
        print("\nWorkspace graph built")
        print("Graph has", self.graph.number_of_nodes(), "nodes")
        print("Graph has", self.graph.number_of_edges(), "edges")

    def add_workspace_node(self, point):
        """Add a workspace node to the graph

        Args:
            point: workspace point to add to the graph
        """
        node_i = self.graph.number_of_nodes()
        self.graph.add_node(node_i, point=point)

    def add_workspace_edge(self, i, j):
        """Add an edge between two workspace nodes

        Args:
            i: index of the first node
            j: index of the second node
        """
        # To ensure the order for the future operation, always have i < j
        if i == j:
            return
        if i > j:
            i, j = j, i

        self.graph.add_edge(
            i,
            j,
            weight=self.robot.workspace_distance(
                self.graph.nodes[i]["point"], self.graph.nodes[j]["point"]
            ),
        )

    def get_workspace_neighbors(self, point, nn, radius=None, k=None):
        """Get nearest neighbors of a point in the workspace

        Args:
            point: point in the workspace
            nn: nearest neighbor search structure
            radius: radius of the neighborhood
            k: number of nearest neighbors to connect to

        Returns:
            neighbors: indices of the nearest neighbors
        """
        point = np.array(point)

        # Nearest neighbors
        # only indices are needed

        # If it is a BallTree
        if isinstance(nn, BallTree):
            if radius is not None:
                return nn.query_radius(
                    [point], radius, return_distance=True, sort_results=True
                )[0][0]

            else:
                return nn.query(
                    [point], k, return_distance=True, sort_results=True
                )[1][0]

        # If it is a NNDescent
        elif isinstance(nn, NNDescent):
            if radius is not None:
                raise ValueError("NNDescent does not support radius search")

            else:
                # Neighbor accuracy is critical for the performance.
                # For higher accuracy, use more neighbors
                new_k = max(k, 200)
                candidates = nn.query([point], k=new_k, epsilon=0.75)[0][0]
                return candidates[:k]

    def visualize_workspace_graph(self):
        """Visualize the workspace graph"""
        # Position

        # Extract node and edge positions
        node_pos = np.array(
            [node["point"][:3] for _, node in self.graph.nodes(data=True)]
        )
        edge_pos = np.array(
            [(node_pos[i], node_pos[j]) for i, j in self.graph.edges()]
        )

        # Plot
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(*node_pos.T, s=10, ec="w", c="g")
        for edge in edge_pos:
            ax.plot(*edge.T, color="y")

        # Show
        ax.grid(True)
        plt.show()

        # Rotation
        if self.robot.rotation != "variable":
            return

        # Extract quaternions
        quats = np.array(
            set(
                [node["point"][3:7] for _, node in self.graph.nodes(data=True)]
            )
        )
        eulers = [quat_to_euler(quat) for quat in quats]

        # Initialize arrays for spherical coordinates
        x, y, z, u, v, w = [], [], [], [], [], []
        for euler in eulers:
            # Convert pitch (euler[1]) and yaw (euler[0])
            # to Cartesian coordinates for the sphere
            roll, pitch, yaw = euler
            x0 = np.cos(pitch) * np.cos(yaw)
            y0 = np.cos(pitch) * np.sin(yaw)
            z0 = np.sin(pitch)

            # Calculate tangent vectors for the sphere at point (x0, y0, z0)
            r_vec = np.array([x0, y0, z0])
            # choose a up vector
            up_vector = np.array([0, 0, 1])
            # avoid singularity by choosing a different up_vector
            if np.allclose(r_vec, up_vector) or np.allclose(r_vec, -up_vector):
                up_vector = np.array([0, 1, 0])
            # cross product with r_vec to get a unit tangent vector
            tan_vec = np.cross(r_vec, up_vector)
            tan_vec /= np.linalg.norm(tan_vec)  # Normalize
            # apply roll around the tangent vector
            rot_matrix = rotvec_to_quat(roll * r_vec)
            vector_rotated = rot_matrix * tan_vec

            # Store for plotting
            x.append(x0)
            y.append(y0)
            z.append(z0)
            u.append(vector_rotated[0])
            v.append(vector_rotated[1])
            w.append(vector_rotated[2])

        # Plot
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # plot points on the sphere
        ax.scatter(x, y, z, color="k", s=5)
        # plot quivers
        for i, _ in enumerate(x):
            ax.quiver(
                x[i],
                y[i],
                z[i],
                u[i],
                v[i],
                w[i],
                color="r",
                length=0.01,
                normalize=True,
            )

        # Draw a sphere for reference
        psi, theta = np.mgrid[
            0 : 2 * np.pi : 100j, -np.pi / 2 : np.pi / 2 : 50j
        ]
        xs = 0.95 * np.cos(theta) * np.cos(psi)
        ys = 0.95 * np.cos(theta) * np.sin(psi)
        zs = 0.95 * np.sin(theta)
        ax.plot_surface(xs, ys, zs, color="w", alpha=0.1)

        # Show
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

        plt.show()

        # Rotation vector representation
        rot_vec = np.array([quat_to_rotvec(q) for q in quats])

        # Plotting
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(rot_vec[:, 0], rot_vec[:, 1], rot_vec[:, 2])

        # Draw a sphere for reference
        psi, theta = np.mgrid[
            0 : 2 * np.pi : 100j, -np.pi / 2 : np.pi / 2 : 50j
        ]
        xs = np.pi * np.cos(theta) * np.cos(psi)
        ys = np.pi * np.cos(theta) * np.sin(psi)
        zs = np.pi * np.sin(theta)
        ax.plot_surface(xs, ys, zs, color="w", alpha=0.1)

        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_zlim([-np.pi, np.pi])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Rotation Vectors Distribution")
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        plt.show()

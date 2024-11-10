"""NearestNeighborsGNAT Data Structure from OMPL
https://ompl.kavrakilab.org/NearestNeighborsGNAT_8h_source.html
"""

import math
import random
from abc import ABC, abstractmethod
import heapq
from collections import deque


# Define the abstract base class for NearestNeighbors
class NearestNeighbors(ABC):
    def __init__(self):
        self.dist_fn = None

    def set_distance_function(self, dist_fn):
        self.dist_fn = dist_fn

    @abstractmethod
    def report_sorted_results(self):
        pass

    @abstractmethod
    def clear(self, data):
        pass

    @abstractmethod
    def add(self, data):
        pass

    def add_list(self, data_list):
        for data in data_list:
            self.add(data)

    @abstractmethod
    def remove(self, data):
        pass

    @abstractmethod
    def nearest(self, data):
        pass

    @abstractmethod
    def nearest_k(self, data, k):
        pass

    @abstractmethod
    def nearest_r(self, data, radius):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def list(self, data_list):
        pass


# Define the GNAT Node class
class GNATNode:
    def __init__(self, degree, capacity, pivot, gnat_sampler=False):
        self.degree = degree
        self.pivot = pivot
        self.min_radius = float("inf")
        self.max_radius = -float("inf")
        self.min_range = [self.min_radius for _ in range(degree)]
        self.max_range = [self.max_radius for _ in range(degree)]
        self.gnat_sampler = gnat_sampler
        if self.gnat_sampler:
            self.sub_tree_size = 1
            self.activity = 0

        self.data = []
        self.children = []

    def update_radius(self, dist):
        if self.min_radius > dist:
            self.min_radius = dist

        if not self.gnat_sampler:
            if self.max_radius < dist:
                self.max_radius = dist
        else:
            if self.max_radius < dist:
                self.max_radius = dist
                self.activity = 0
            else:
                self.activity = max(-32, self.activity - 1)

    def update_range(self, i, dist):
        if self.min_range[i] > dist:
            self.min_range[i] = dist
        if self.max_range[i] < dist:
            self.max_range[i] = dist

    def add(self, gnat, data):
        gnat: GNAT

        if self.gnat_sampler:
            self.sub_tree_size += 1

        if len(self.children) == 0:
            self.data.append(data)
            gnat.size += 1
            if self.need_to_split(gnat):
                if len(gnat.removed) > 0:
                    gnat.rebuild_data_structure()
                elif gnat.size >= gnat.rebuild_size:
                    gnat.rebuild_size *= 2
                    gnat.rebuild_data_structure()
                else:
                    self.split(gnat)
        else:
            dist = [0] * len(self.children)
            dist[0] = gnat.dist_fn(data, self.children[0].pivot)
            min_dist = dist[0]
            min_index = 0

            for i in range(1, len(self.children)):
                dist[i] = gnat.dist_fn(data, self.children[i].pivot_)
                if dist[i] < self.min_range[min_index]:
                    min_dist = dist[i]
                    min_index = i
            for i in range(len(self.children)):
                self.children[i].update_range(min_index, dist[i])
            self.children[min_index].update_radius(min_dist)
            self.children[min_index].add(gnat, data)

    def need_to_split(self, gnat):
        gnat: GNAT

        sz = len(self.data)
        return sz > gnat.max_num_pts_per_leaf and sz > self.degree

    # TODO
    def split(self, gnat):
        dists = [
            [0.0 for _ in range(self.degree)] for _ in range(len(self.data))
        ]
        pivots = []

        self.children = []
        gnat.pivot_selector.kcenters(self.data, self.degree, pivots, dists)

        for pivot_index in pivots:
            child_node = GNATNode(
                self.degree,
                gnat.max_num_pts_per_leaf,
                self.data[pivot_index],
                self.gnat_sampler,
            )
            self.children.append(child_node)

        self.degree = len(pivots)

        for j in range(len(self.data)):
            k = 0
            for i in range(1, self.degree):
                if dists[j][i] < dists[j][k]:
                    k = i
            child = self.children[k]
            if j != pivots[k]:
                child.data.append(self.data[j])
                child.update_radius(dists[j][k])
            for i in range(self.degree):
                self.children[i].update_range(k, dists[j][i])

        for child in self.children:
            child.degree = min(
                max(
                    int((self.degree * len(child.data)) / len(self.data)),
                    gnat.min_degree,
                ),
                gnat.max_degree,
            )
            if child.min_radius >= float("inf"):
                child.min_radius = child.max_radius = 0.0
            if self.gnat_sampler:
                child.sub_tree_size = len(child.data) + 1

        self.data = []

        for child in self.children:
            if child.need_to_split(gnat):
                child.split(gnat)

    def insert_neighbor_k(self, nbh, k, data, key, dist):
        if len(nbh) < k:
            heapq.heappush(nbh, (-dist, data))
            return True
        elif dist < -nbh[0][0] or (
            dist < sys.float_info.epsilon and data == key
        ):
            heapq.heappop(nbh)
            heapq.heappush(nbh, (-dist, data))
            return True
        return False

    def nearest_k(self, gnat, data, k, nbh, node_queue, is_pivot):
        for d in self.data:
            if not gnat.is_removed(d):
                if self.insert_neighbor_k(
                    nbh, k, d, data, gnat.dist_fn(data, d)
                ):
                    is_pivot[0] = False
        if len(self.children) > 0:
            sz = len(self.children)
            offset = gnat.offset
            gnat.offset += 1
            dist_to_pivot = [0.0] * sz
            permutation = [0] * sz
            for i in range(sz):
                permutation[i] = (i + offset) % sz

            for i in range(sz):
                if permutation[i] >= 0:
                    p_i = permutation[i]
                    child = self.children[p_i]
                    dist_to_pivot[p_i] = gnat.dist_fn(data, child.pivot)
                    if self.insert_neighbor_k(
                        nbh, k, child.pivot, data, dist_to_pivot[p_i]
                    ):
                        is_pivot[0] = True
                    if len(nbh) == k:
                        dist = -nbh[0][0]
                        for j in range(sz):
                            if permutation[j] >= 0 and i != j:
                                p_j = permutation[j]
                                if (
                                    dist_to_pivot[p_i] - dist
                                    > child.max_range[p_j]
                                    or dist_to_pivot[p_i] + dist
                                    < child.min_range[p_j]
                                ):
                                    permutation[j] = -1
            if len(nbh) > 0:
                dist = -nbh[0][0]
            else:
                dist = float("inf")
            for p in permutation:
                if p >= 0:
                    child = self.children[p]
                    if len(nbh) < k or (
                        dist_to_pivot[p] - dist <= child.max_radius
                        and dist_to_pivot[p] + dist >= child.min_radius
                    ):
                        heapq.heappush(node_queue, (dist_to_pivot[p], child))

    def insert_neighbor_r(self, nbh, r, data, dist):
        if dist <= r:
            heapq.heappush(nbh, (dist, data))

    def nearest_r(self, gnat, data, r, nbh, node_queue):
        dist = r
        for d in self.data:
            if not gnat.is_removed(d):
                self.insert_neighbor_r(nbh, r, d, gnat.dist_fn(data, d))
        if len(self.children) > 0:
            sz = len(self.children)
            offset = gnat.offset
            gnat.offset += 1
            dist_to_pivot = [0.0] * sz
            permutation = [0] * sz
            for i in range(sz):
                permutation[i] = (i + offset) % sz

            for i in range(sz):
                if permutation[i] >= 0:
                    p_i = permutation[i]
                    child = self.children[p_i]
                    dist_to_pivot[p_i] = gnat.dist_fn(data, child.pivot)
                    self.insert_neighbor_r(
                        nbh, r, child.pivot, dist_to_pivot[p_i]
                    )
                    for j in range(sz):
                        if permutation[j] >= 0 and i != j:
                            p_j = permutation[j]
                            if (
                                dist_to_pivot[p_i] - dist
                                > child.max_range[p_j]
                                or dist_to_pivot[p_i] + dist
                                < child.min_range[p_j]
                            ):
                                permutation[j] = -1
            for p in permutation:
                if p >= 0:
                    child = self.children[p]
                    if (
                        dist_to_pivot[p] - dist <= child.max_radius
                        and dist_to_pivot[p] + dist >= child.min_radius
                    ):
                        heapq.heappush(node_queue, (dist_to_pivot[p], child))

    def list(self, gnat, data_list):
        if not gnat.is_removed(self.pivot):
            data_list.append(self.pivot)
        for d in self.data:
            if not gnat.is_removed(d):
                data_list.append(d)
        for child in self.children:
            child.list(gnat, data_list)

    if (
        hasattr(sys.modules[__name__], "GNAT_SAMPLER")
        and sys.modules[__name__].GNAT_SAMPLER
    ):

        def get_sampling_weight(self, gnat):
            min_r = float("inf")
            for min_range in self.min_range:
                if min_range < min_r and min_range > 0.0:
                    min_r = min_range
            min_r = max(min_r, self.max_radius)
            return (min_r**gnat.estimated_dimension) / float(
                self.sub_tree_size
            )

        def sample(self, gnat, rng):
            if len(self.children) != 0:
                if rng.uniform(0, 1) < 1.0 / float(self.sub_tree_size):
                    return self.pivot
                distribution = []
                weights = []
                for child in self.children:
                    weight = child.get_sampling_weight(gnat)
                    distribution.append(child)
                    weights.append(weight)
                selected_child = random.choices(
                    distribution, weights=weights, k=1
                )[0]
                return selected_child.sample(gnat, rng)
            else:
                i = rng.randint(0, len(self.data))
                if i == len(self.data):
                    return self.pivot
                else:
                    return self.data[i]


class GNAT(NearestNeighbors):
    def __init__(
        self,
        max_degree=4,
        min_degree=2,
        max_radius=float("inf"),
        min_radius=0.0,
        max_level=float("inf"),
    ):
        self.max_degree = max_degree
        self.min_degree = min_degree
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.max_level = max_level
        self.root = None

    class Node:
        def __init__(self, pivot):
            self.pivot = pivot
            self.children = []
            self.points = []
            self.parent = None
            self.radius = 0.0
            self.coverage = 0.0
            self.dmin = {}
            self.dmax = {}

    def distance(self, a, b):
        # User-defined distance function for SE(3)
        # Implement your SE(3) distance metric here
        # Placeholder: Euclidean distance for translation components
        translation_distance = math.sqrt(
            sum(
                (ai - bi) ** 2
                for ai, bi in zip(a["translation"], b["translation"])
            )
        )
        # Rotation distance can be added here
        rotation_distance = 0.0  # Implement rotation distance
        return translation_distance + rotation_distance

    def insert(self, point):
        if self.root is None:
            self.root = self.Node(point)
        else:
            self._insert(self.root, point, 0)

    def _insert(self, node, point, level):
        if level >= self.max_level:
            node.points.append(point)
            return

        dist = self.distance(point, node.pivot)

        # Update radius and coverage
        node.radius = max(node.radius, dist)
        node.coverage = max(node.coverage, dist)

        # Find the best child to insert the point
        best_child = None
        min_dist = float("inf")

        for child in node.children:
            d = self.distance(point, child.pivot)
            if d < min_dist:
                min_dist = d
                best_child = child

        if best_child and min_dist <= best_child.coverage:
            self._insert(best_child, point, level + 1)
        else:
            # Create a new child node
            new_node = self.Node(point)
            new_node.parent = node
            node.children.append(new_node)

            # Update dmin and dmax for all siblings
            for child in node.children:
                if child == new_node:
                    continue
                d = self.distance(new_node.pivot, child.pivot)
                new_node.dmin[child] = d
                new_node.dmax[child] = d
                child.dmin[new_node] = d
                child.dmax[new_node] = d

            # Enforce max degree
            if len(node.children) > self.max_degree:
                self._split_node(node, level)

    def _split_node(self, node, level):
        # Implement node splitting logic if necessary
        pass  # For simplicity, splitting is not implemented here

    def nearest_neighbor(self, query_point, k=1):
        best = []
        self._search(self.root, query_point, best, k)
        return [item[1] for item in sorted(best)[:k]]

    def _search(self, node, query_point, best, k):
        if node is None:
            return

        dist = self.distance(query_point, node.pivot)

        # Update best candidates
        if len(best) < k or dist < best[-1][0]:
            best.append((dist, node.pivot))
            best.sort(key=lambda x: x[0])
            if len(best) > k:
                best.pop()

        # Calculate pruning bounds
        for child in node.children:
            d = self.distance(query_point, child.pivot)
            min_possible = d - child.coverage
            max_possible = d + child.coverage

            if len(best) < k or min_possible < best[-1][0]:
                self._search(child, query_point, best, k)

        # Search points at this node
        for point in node.points:
            d = self.distance(query_point, point)
            if len(best) < k or d < best[-1][0]:
                best.append((d, point))
                best.sort(key=lambda x: x[0])
                if len(best) > k:
                    best.pop()

    def remove(self, point):
        self._remove(self.root, point)

    def _remove(self, node, point):
        if node is None:
            return False

        if point == node.pivot:
            # Remove pivot point
            if node.parent:
                node.parent.children.remove(node)
            else:
                self.root = None
            return True

        # Check children
        for child in node.children:
            if self._remove(child, point):
                return True

        # Check points at this node
        if point in node.points:
            node.points.remove(point)
            return True

        return False


# Example usage:

# Define sample SE(3) points
points = [
    {"translation": [1, 2, 3], "rotation": [1, 0, 0, 0]},
    {"translation": [4, 5, 6], "rotation": [0, 1, 0, 0]},
    {"translation": [7, 8, 9], "rotation": [0, 0, 1, 0]},
    {"translation": [2, 3, 4], "rotation": [0, 0, 0, 1]},
    # Add more points as needed
]

# Create GNAT instance
gnat = GNAT(max_degree=3, min_degree=1)

# Insert points into GNAT
for point in points:
    gnat.insert(point)

# Query the nearest neighbor
query_point = {"translation": [3, 3, 3], "rotation": [1, 0, 0, 0]}
nearest = gnat.nearest_neighbor(query_point, k=1)
print("Nearest neighbor:", nearest)

# Remove a point
gnat.remove(points[0])

"""NearestNeighborsGNAT Data Structure from OMPL
Original source: https://ompl.kavrakilab.org/NearestNeighborsGNAT_8h_source.html
Original authors: Mark Moll, Bryant Gipson

Implemented in Python by Zhuoyun Zhong
The script now strictly follows the original C++ implementation
and is not yet optimized for Python.
"""

import sys
import random
import heapq
import numpy as np
from typing import List, Tuple, Callable, Hashable
from .nearest_neighbors import NearestNeighbors, GreedyKCenters


# Define the GNAT class
class GNAT(NearestNeighbors):
    def __init__(
        self,
        degree=8,
        min_degree=4,
        max_degree=12,
        max_num_pts_per_leaf=50,
        removed_cache_size=500,
        rebalancing=False,
        estimated_dimension=6.0,
        gnat_sampler=False,
    ):
        super().__init__()

        self.tree: GNATNode | None = None
        self.degree = degree
        self.min_degree = min(degree, min_degree)
        self.max_degree = max(max_degree, degree)
        self.max_num_pts_per_leaf = max_num_pts_per_leaf
        self.data_size = 0
        self.rebuild_size = (
            max_num_pts_per_leaf * degree if rebalancing else sys.maxsize
        )
        self.removed_cache_size = removed_cache_size
        self.pivot_selector = GreedyKCenters()
        self.removed = set()
        # used to cycle through children of a node in different orders
        self.offset = 0

        self.estimated_dimension = estimated_dimension
        self.gnat_sampler = gnat_sampler

    def set_distance_function(
        self, dist_fn: Callable[[object, object], float]
    ):
        super().set_distance_function(dist_fn)
        self.pivot_selector.set_distance_function(dist_fn)
        if self.tree:
            self.rebuild_data_structure()

    def clear(self):
        if self.tree:
            self.tree = None
        self.data_size = 0
        self.removed.clear()
        if self.rebuild_size != sys.maxsize:
            self.rebuild_size = self.max_num_pts_per_leaf * self.degree

    def report_sorted_results(self) -> bool:
        return True

    def add(self, data: object):
        if not isinstance(data, Hashable):
            raise Exception("Input data must be hashable")

        if self.tree:
            if self.is_removed(data):
                self.rebuild_data_structure()
            self.tree.add(self, data)
        else:
            self.tree = GNATNode(
                self.degree, self.max_num_pts_per_leaf, data, self.gnat_sampler
            )
            self.data_size = 1

    def add_list(self, data_list: List[object]):
        if not isinstance(data_list[0], Hashable):
            raise Exception("Input data must be hashable")

        if self.tree:
            super().add_list(data_list)

        elif len(data_list) != 0:
            self.tree = GNATNode(
                self.degree,
                self.max_num_pts_per_leaf,
                data_list[0],
                self.gnat_sampler,
            )

            if self.gnat_sampler:
                self.tree.sub_tree_size = len(data_list)

            self.tree.data.extend(data_list[1:])
            self.data_size += len(data_list)
            if self.tree.need_to_split(self):
                self.tree.split(self)

    def rebuild_data_structure(self):
        data_list = []
        self.list(data_list)
        self.clear()
        self.add_list(data_list)

    def remove(self, data: object):
        if self.data_size == 0:
            return False

        # find data in tree
        is_pivot, nbh_queue = self.nearest_k_internal(data, 1)
        if not nbh_queue:
            return False
        d = nbh_queue[0][1]
        if d != data:
            return False
        self.removed.add(d)
        self.data_size -= 1

        # if we removed a pivot or if the capacity of removed elements
        # has been reached, we rebuild the entire GNAT
        if is_pivot or len(self.removed) >= self.removed_cache_size:
            self.rebuild_data_structure()
        return True

    def nearest(self, data) -> object:
        if self.data_size:
            is_pivot, nbh_queue = self.nearest_k_internal(data, 1)
            if nbh_queue:
                return nbh_queue[0][1]
        raise Exception(
            "No elements found in nearest neighbors data structure"
        )

    def nearest_k(self, data, k) -> List[object]:
        nbh = []
        if k == 0:
            return nbh
        if self.data_size:
            is_pivot, nbh_queue = self.nearest_k_internal(data, k)
            nbh = self.postprocess_nearest(nbh_queue)
        return nbh

    def nearest_r(self, data, radius) -> List[object]:
        nbh = []
        if self.data_size:
            nbh_queue = self.nearest_r_internal(data, radius)
            nbh = self.postprocess_nearest(nbh_queue)
        return nbh

    def size(self) -> int:
        return self.data_size

    def sample(self) -> object:
        if not self.data_size():
            raise Exception("Cannot sample from an empty tree")
        else:
            return self.tree.sample(self)

    def list(self, data_list):
        data_list.clear()
        if self.tree:
            self.tree.list(self, data_list)

    def is_removed(self, data) -> bool:
        return data in self.removed

    def nearest_k_internal(
        self, data: object, k: int
    ) -> Tuple[bool, List[Tuple[float, object]]]:
        nbh_queue = []

        dist = self.dist_fn(data, self.tree.pivot)
        is_pivot = self.tree.insert_neighbor_k(
            nbh_queue, k, self.tree.pivot, data, dist
        )

        node_queue = []
        is_pivot = self.tree.nearest_k(
            self, data, k, nbh_queue, node_queue, is_pivot
        )
        while node_queue:
            dist = -nbh_queue[0][0]  # Max-heap
            node_dist = heapq.heappop(node_queue)
            node = node_dist[1]
            dist_to_pivot = node_dist[2]
            if len(nbh_queue) == k and (
                dist_to_pivot > node.max_radius + dist
                or dist_to_pivot < node.min_radius - dist
            ):
                continue
            is_pivot = node.nearest_k(
                self, data, k, nbh_queue, node_queue, is_pivot
            )
        return is_pivot, nbh_queue

    def nearest_r_internal(
        self, data: object, radius: float
    ) -> List[Tuple[float, object]]:
        nbh_queue = []
        dist = radius

        node_queue = []
        self.tree.insert_neighbor_r(
            nbh_queue,
            radius,
            self.tree.pivot,
            self.dist_fn(data, self.tree.pivot),
        )
        self.tree.nearest_r(self, data, radius, nbh_queue, node_queue)
        while node_queue:
            node_dist = heapq.heappop(node_queue)
            node = node_dist[1]
            dist_to_pivot = node_dist[2]
            if (
                dist_to_pivot > node.max_radius + dist
                or dist_to_pivot < node.min_radius - dist
            ):
                continue
            node.nearest_r(self, data, radius, nbh_queue, node_queue)

        return nbh_queue

    def postprocess_nearest(self, nbh_queue: List[object]) -> List[object]:
        nbh = []
        while nbh_queue:
            nbh.append(heapq.heappop(nbh_queue)[1])
        nbh.reverse()  # in ascending order
        return nbh


# Define the GNAT Node class
class GNATNode:
    def __init__(
        self,
        degree: int,
        capacity: int,
        pivot: object,
        gnat_sampler: bool = False,
    ):
        self.degree = degree
        self.capacity = capacity  # not used
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

    def update_radius(self, dist: float):
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

    def update_range(self, i: int, dist: float):
        if self.min_range[i] > dist:
            self.min_range[i] = dist
        if self.max_range[i] < dist:
            self.max_range[i] = dist

    def add(self, gnat: GNAT, data: object):
        if self.gnat_sampler:
            self.sub_tree_size += 1

        if len(self.children) == 0:
            self.data.append(data)
            gnat.data_size += 1
            if self.need_to_split(gnat):
                if len(gnat.removed) > 0:
                    gnat.rebuild_data_structure()
                elif gnat.data_size >= gnat.rebuild_size:
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
                dist[i] = gnat.dist_fn(data, self.children[i].pivot)
                if dist[i] < min_dist:
                    min_dist = dist[i]
                    min_index = i
            for i in range(len(self.children)):
                self.children[i].update_range(min_index, dist[i])
            self.children[min_index].update_radius(min_dist)
            self.children[min_index].add(gnat, data)

    def need_to_split(self, gnat: GNAT) -> bool:
        sz = len(self.data)
        return sz > gnat.max_num_pts_per_leaf and sz > self.degree

    def split(self, gnat: GNAT):
        self.children = []

        pivots, dists = gnat.pivot_selector.kcenters(self.data, self.degree)
        for pivot in pivots:
            child_node = GNATNode(
                self.degree,
                gnat.max_num_pts_per_leaf,
                self.data[pivot],
                self.gnat_sampler,
            )
            self.children.append(child_node)

        # in case fewer than degree_ pivots were found
        self.degree = len(pivots)

        for j in range(len(self.data)):
            k = 0
            for i in range(1, self.degree):
                if dists[j, i] < dists[j, k]:
                    k = i
            child = self.children[k]
            if j != pivots[k]:
                child.data.append(self.data[j])
                child.update_radius(dists[j, k])
            for i in range(self.degree):
                self.children[i].update_range(k, dists[j, i])

        for child in self.children:
            # make sure degree lies between min_degree and max_degree
            child.degree = min(
                max(
                    self.degree * len(child.data) // len(self.data),
                    gnat.min_degree,
                ),
                gnat.max_degree,
            )
            # singleton
            if child.min_radius >= float("inf"):
                child.min_radius = 0.0
                child.max_radius = 0.0
            # set subtree size
            if self.gnat_sampler:
                child.sub_tree_size = len(child.data) + 1

        self.data = []
        # check if new leaves need to be split
        for child in self.children:
            if child.need_to_split(gnat):
                child.split(gnat)

    def insert_neighbor_k(
        self,
        nbh: List[Tuple[float, object]],
        k: int,
        data: object,
        key: object,
        dist: float,
    ) -> bool:
        if len(nbh) < k:
            heapq.heappush(nbh, (-dist, data))  # max heap
            return True
        elif (dist < -nbh[0][0]) or (
            dist < sys.float_info.epsilon and data == key
        ):
            heapq.heappop(nbh)
            heapq.heappush(nbh, (-dist, data))  # max heap
            return True
        return False

    def nearest_k(
        self,
        gnat: GNAT,
        data: object,
        k: int,
        nbh: List[Tuple[float, object]],  # max heap
        node_queue: List[Tuple[float, "GNATNode", float]],  # max heap
        is_pivot: bool,
    ) -> bool:
        for d in self.data:
            if not gnat.is_removed(d):
                if self.insert_neighbor_k(
                    nbh, k, d, data, gnat.dist_fn(data, d)
                ):
                    is_pivot = False

        if len(self.children) <= 0:
            return is_pivot

        sz = len(self.children)
        offset = gnat.offset
        gnat.offset += 1
        dist_to_pivot = [0.0] * sz
        permutation = [(i + offset) % sz for i in range(sz)]

        for i in range(sz):
            if permutation[i] < 0:
                continue

            p_i = permutation[i]
            child = self.children[p_i]
            dist_to_pivot[p_i] = gnat.dist_fn(data, child.pivot)

            if self.insert_neighbor_k(
                nbh, k, child.pivot, data, dist_to_pivot[p_i]
            ):
                is_pivot = True

            if len(nbh) == k:
                dist = -nbh[0][0]  # from a max-heap
                for j in range(sz):
                    if permutation[j] >= 0 and i != j:
                        p_j = permutation[j]
                        if (
                            dist_to_pivot[p_i] - dist > child.max_range[p_j]
                            or dist_to_pivot[p_i] + dist < child.min_range[p_j]
                        ):
                            permutation[j] = -1

        dist = -nbh[0][0]
        for p in permutation:
            if p < 0:
                continue

            child = self.children[p]
            if len(nbh) < k or (
                dist_to_pivot[p] - dist <= child.max_radius
                and dist_to_pivot[p] + dist >= child.min_radius
            ):
                metric = dist_to_pivot[p] - child.max_radius
                heapq.heappush(
                    node_queue, (-metric, child, dist_to_pivot[p])
                )  # max heap

        return is_pivot

    def insert_neighbor_r(
        self,
        nbh: List[Tuple[float, object]],
        r: float,
        data: object,
        dist: float,
    ):
        if dist <= r:
            heapq.heappush(nbh, (-dist, data))  # max heap

    def nearest_r(
        self,
        gnat: GNAT,
        data: object,
        r: float,
        nbh: List[Tuple[float, object]],  # max heap
        node_queue: List[Tuple[float, "GNATNode", float]],  # max heap
    ):
        dist = r
        for d in self.data:
            if not gnat.is_removed(d):
                self.insert_neighbor_r(nbh, r, d, gnat.dist_fn(data, d))

        if len(self.children) <= 0:
            return

        sz = len(self.children)
        offset = gnat.offset
        gnat.offset += 1
        dist_to_pivot = [0.0] * sz
        permutation = [(i + offset) % sz for i in range(sz)]

        for i in range(sz):
            if permutation[i] < 0:
                continue

            p_i = permutation[i]
            child = self.children[p_i]
            dist_to_pivot[p_i] = gnat.dist_fn(data, child.pivot)
            self.insert_neighbor_r(nbh, r, child.pivot, dist_to_pivot[p_i])
            for j in range(sz):
                if permutation[j] >= 0 and i != j:
                    p_j = permutation[j]
                    if (
                        dist_to_pivot[p_i] - dist > child.max_range[p_j]
                        or dist_to_pivot[p_i] + dist < child.min_range[p_j]
                    ):
                        permutation[j] = -1

        for p in permutation:
            if p < 0:
                continue

            child = self.children[p]
            if (
                dist_to_pivot[p] - dist <= child.max_radius
                and dist_to_pivot[p] + dist >= child.min_radius
            ):
                metric = dist_to_pivot[p] - child.max_radius
                heapq.heappush(
                    node_queue, (-metric, child, dist_to_pivot[p])
                )  # max heap

    def list(self, gnat: GNAT, data_list: List[object]):
        if not gnat.is_removed(self.pivot):
            data_list.append(self.pivot)

        for d in self.data:
            if not gnat.is_removed(d):
                data_list.append(d)

        for child in self.children:
            child.list(gnat, data_list)

    def get_sampling_weight(self, gnat: GNAT) -> float:
        min_r = float("inf")
        for min_range in self.min_range:
            if min_range < min_r and min_range > 0.0:
                min_r = min_range
        min_r = max(min_r, self.max_radius)
        return (min_r**gnat.estimated_dimension) / self.sub_tree_size

    def sample(self, gnat: GNAT) -> object:
        if len(self.children) != 0:
            if random.uniform(0, 1) < 1.0 / self.sub_tree_size:
                return self.pivot
            distribution = []
            weights = []
            for child in self.children:
                distribution.append(child)
                weights.append(child.get_sampling_weight(gnat))
            selected_child = random.choices(
                distribution, weights=weights, k=1
            )[0]
            return selected_child.sample(gnat)

        else:
            i = random.randint(0, len(self.data))
            if i == len(self.data):
                return self.pivot
            else:
                return self.data[i]


if __name__ == "__main__":
    import time
    import math
    from sklearn.neighbors import BallTree
    from scipy.spatial.transform import Rotation as R
    from .utils import se3_distance

    random.seed(42)
    np.random.seed(42)

    # Try to test accuracy and performance by comparing with Sklearn's BallTree

    # Define a simple distance function for SE3 points
    def euclidean_distance(p1, p2, w=1.0):
        # p_diff = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        # q_diff = abs((p1[2] - p2[2] + np.pi) % (2 * np.pi) - np.pi)
        # return p_diff + w * q_diff

        d_position = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        d_rotation = 1 - np.abs(np.dot(p1[3:7], p2[3:7]))
        return d_position + w * d_rotation

    # Create a set of random 2D points
    data_points = [
        (
            random.uniform(0, 100),
            random.uniform(0, 100),
            random.uniform(0, 100),
            *R.random().as_quat(),
        )
        for _ in range(1000000)
    ]

    # Initialize GNAT
    start_time = time.time()
    gnat = GNAT()
    gnat.set_distance_function(euclidean_distance)
    # Add points to GNAT
    gnat.add_list(data_points)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.6f} seconds")

    # Test nearest neighbor search
    query_point = (50, 50, 50, 0, 0, 0, 1)
    nearest = gnat.nearest(query_point)
    print(f"Nearest neighbor to {query_point} is {nearest}")

    # Test k-nearest neighbors search
    start_time = time.time()
    k = 5
    nearest_k = gnat.nearest_k(query_point, k)
    search_time = time.time() - start_time
    print(f"Nearest neighbor search time: {search_time:.6f} seconds")
    print(f"{k} nearest neighbors to {query_point} are:")
    for neighbor in nearest_k:
        print(neighbor)

    # # Test range search
    # radius = 1
    # nearest_r = gnat.nearest_r(query_point, radius)
    # print(f"Neighbors within radius {radius} of {query_point} are:")
    # for neighbor in nearest_r:
    #     print(neighbor)

    # See the Balltree results for comparison
    print("\nBallTree")
    start_time = time.time()
    ball_tree = BallTree(data_points, metric=euclidean_distance)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.6f} seconds")

    # Convert query point to numpy array
    query_array = np.array([query_point])

    # Use BallTree to find the nearest neighbor
    distances, indices = ball_tree.query(query_array, k=1)
    ball_tree_nearest = data_points[indices[0][0]]
    print(f"Nearest neighbor to {query_point}: {tuple(ball_tree_nearest)}")

    # Use BallTree to find k-nearest neighbors
    start_time = time.time()
    distances, indices = ball_tree.query(query_array, k=k)
    search_time = time.time() - start_time
    print(f"Nearest neighbor search time: {search_time:.6f} seconds")
    ball_tree_nearest_k = [data_points[i] for i in indices[0]]
    print(f"{k} nearest neighbors to {query_point}:")
    for neighbor in ball_tree_nearest_k:
        print(tuple(neighbor))

    # # Use BallTree to find all neighbors within the given radius
    # indices = ball_tree.query_radius(query_array, r=radius)
    # ball_tree_range_search = [data_points[i] for i in indices[0]]
    # print(f"Neighbors within radius {radius} of {query_point}:")
    # for neighbor in ball_tree_range_search:
    #     print(tuple(neighbor))

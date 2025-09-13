"""NearestNeighborsGNAT Data Structure from OMPL
Original source: 
https://ompl.kavrakilab.org/NearestNeighbors_8h_source.html
https://ompl.kavrakilab.org/classompl_1_1GreedyKCenters.html
Original authors: Mark Moll, Ioan Sucan

Implemented in Python by Zhuoyun Zhong
The script now strictly follows the original C++ implementation
and is not yet optimized for Python.
"""

import sys
import random
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable


# Define the abstract base class for NearestNeighbors
class NearestNeighbors(ABC):
    def __init__(self):
        self.dist_fn = None

    def set_distance_function(
        self, dist_fn: Callable[[object, object], float]
    ):
        self.dist_fn = dist_fn

    @abstractmethod
    def report_sorted_results(self) -> bool:
        pass

    @abstractmethod
    def clear(self, data: object):
        pass

    @abstractmethod
    def add(self, data: object):
        pass

    def add_list(self, data_list: List[object]):
        for data in data_list:
            self.add(data)

    @abstractmethod
    def remove(self, data: object):
        pass

    @abstractmethod
    def nearest(self, data: object) -> object:
        pass

    @abstractmethod
    def nearest_k(self, data: object, k: int) -> List[object]:
        pass

    @abstractmethod
    def nearest_r(self, data: object, radius: float) -> List[object]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def list(self, data_list: List[object]):
        pass


class GreedyKCenters:
    def __init__(self):
        self.dist_fn = None

    def set_distance_function(
        self, dist_fn: Callable[[object, object], float]
    ):
        self.dist_fn = dist_fn

    def kcenters(
        self, data_list: List[object], k: int
    ) -> Tuple[List[int], NDArray]:
        # array containing the minimum distance between each data point
        # and the centers computed so far
        min_dist = [float("inf")] * len(data_list)

        centers = []
        dists = np.zeros((len(data_list), k))

        # First center is picked randomly
        centers.append(random.randint(0, len(data_list) - 1))
        for i in range(1, k):
            ind = 0
            center = data_list[centers[i - 1]]
            max_dist = -float("inf")

            for j in range(len(data_list)):
                dists[j, i - 1] = self.dist_fn(data_list[j], center)
                if dists[j, i - 1] < min_dist[j]:
                    min_dist[j] = dists[j, i - 1]
                # The j-th center is the one furthest away from centers 0,..,i-1
                if min_dist[j] > max_dist:
                    ind = j
                    max_dist = min_dist[j]
            # No more centers available
            if max_dist < sys.float_info.epsilon:
                break
            centers.append(ind)

        center = data_list[centers[-1]]
        i = len(centers) - 1
        for j in range(len(data_list)):
            dists[j, i] = self.dist_fn(data_list[j], center)

        return centers, dists

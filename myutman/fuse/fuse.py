import abc
from typing import List

import numpy as np

from myutman.streaming_algo.streaming_algo import StreamingAlgo


class Fuse:
    def __init__(self):
        self.thresholds = None
        self.stat = None

    @abc.abstractmethod
    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        pass

    def __call__(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        return self.fuse(p, nodes)


class FuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self):
        super(FuseForWindowAlgo, self).__init__()
        self.count = 0

    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        self.count += 1
        stats = [node.get_stat() for node in nodes]
        self.stat = np.max(stats, axis=0)
        self.thresholds = np.quantile(self.stat[:, 1:], 1 - p, axis=-1)
        ans = np.any(self.thresholds < self.stat[:, 0] - FuseForWindowAlgo.EPS)
        # if ans:
        #    print(np.quantile(stats[0][0][1:], 0.99))
        return ans


class MeanFuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self):
        super(MeanFuseForWindowAlgo, self).__init__()
        self.count = 0

    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        self.count += 1
        stats = [node.get_stat() for node in nodes]
        self.stat = np.mean(stats, axis=0)
        self.thresholds = np.quantile(self.stat[:, 1:], 1 - p, axis=-1)
        ans = np.any(self.thresholds < self.stat[:, 0] - FuseForWindowAlgo.EPS)
        # if ans:
        #    print(np.quantile(stats[0][0][1:], 0.99))
        return ans

class MedianFuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self):
        super(MedianFuseForWindowAlgo, self).__init__()
        self.count = 0

    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        self.count += 1
        stats = [node.get_stat() for node in nodes]
        self.stat = np.median(stats, axis=0)
        self.thresholds = np.quantile(self.stat[:, 1:], 1 - p, axis=-1)
        ans = np.any(self.thresholds < self.stat[:, 0] - FuseForWindowAlgo.EPS)
        # if ans:
        #    print(np.quantile(stats[0][0][1:], 0.99))
        return ans
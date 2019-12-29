import abc
from typing import List

import numpy as np

from myutman.single_thread import StreamingAlgo


class Fuse:
    @abc.abstractmethod
    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        pass

    def __call__(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        return self.fuse(p, nodes)


class FuseForWindowAlgo(Fuse):
    def __init__(self):
        super(FuseForWindowAlgo, self).__init__()

    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        stats = [node.get_stat() for node in nodes]
        stat = np.max(stats, axis=0)
        thresholds = np.quantile(stat[:, 1:], 1 - p, axis=-1)
        return np.any(thresholds < stat[:, 0])

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
    EPS = 1e-9

    def __init__(self):
        super(FuseForWindowAlgo, self).__init__()
        self.count = 0

    def fuse(self, p: float, nodes: List[StreamingAlgo]) -> bool:
        self.count += 1
        stats = [node.get_stat() for node in nodes]
        stat = np.max(stats, axis=0)
        thresholds = np.quantile(stat[:, 1:], 1 - p, axis=-1)
        ans = np.any(thresholds < stat[:, 0] - FuseForWindowAlgo.EPS)
        # if ans:
        #    print(np.quantile(stats[0][0][1:], 0.99))
        return ans

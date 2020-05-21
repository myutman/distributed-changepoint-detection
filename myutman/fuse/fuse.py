import abc
import os
from typing import List, Union, Tuple

import numpy as np

from myutman.streaming_algo.streaming_algo import StreamingAlgo


class Fuse:
    def __init__(self, p: float, n_nodes: int):
        self.p = p
        self.n_nodes = n_nodes
        self.thresholds = None
        self.stat = None

    @abc.abstractmethod
    def fuse(self, stats: Union[List[List[float]], np.ndarray]) -> bool:
        pass

    def __call__(self, stats: Union[List[List[float]], np.ndarray]) -> bool:
        return self.fuse(stats)


class FuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self, p: float, n_nodes: int, n_iter: int, window_sizes: List[Tuple[int, int]]):
        super(FuseForWindowAlgo, self).__init__(p, n_nodes)
        self.count = 0
        rnd = np.random.RandomState(0)
        vec = np.load(os.path.join(
            os.path.dirname(__file__),
            f'../precalc_qunatiles/precalced_quantiles_{n_iter}_iter_with_burn-in.npy'
        ))[[[200, 400, 800, 1600].index(s) for s, _ in window_sizes]]
        vecs = [
            [rnd.permutation(sub_vec) for sub_vec in vec] for _ in range(n_nodes)
        ]
        fused_vec = np.max(vecs, axis=0)
        self.thresholds = np.quantile(fused_vec, 1 - p, axis=-1)

    def fuse(self, stats: Union[List[List[float]], np.ndarray]) -> bool:
        self.count += 1
        self.stat = np.max(stats, axis=0)
        ans = np.any(self.thresholds < self.stat - FuseForWindowAlgo.EPS)

        # TODO: remove debug
        if ans:
            win_sizes = [size for size, threshold, stat in zip([200, 400, 800, 1600], self.thresholds, self.stat) if stat > threshold]
            print(f"detected by sizes {win_sizes}")
            if len(win_sizes) == 1:
                print("wow")
        return ans


class MeanFuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self, p: float, n_nodes: int, n_iter: int, window_sizes: List[Tuple[int, int]]):
        super(MeanFuseForWindowAlgo, self).__init__(p, n_nodes)
        self.count = 0
        rnd = np.random.RandomState(0)
        vec = np.load(os.path.join(
            os.path.dirname(__file__),
            f'../precalc_qunatiles/precalced_quantiles_{n_iter}_iter_with_burn-in.npy'
        ))[[[200, 400, 800, 1600].index(s) for s, _ in window_sizes]]
        vecs = [
            [rnd.permutation(sub_vec) for sub_vec in vec] for _ in range(n_nodes)
        ]
        fused_vec = np.mean(vecs, axis=0)
        self.thresholds = np.quantile(fused_vec, 1 - p, axis=-1)

    def fuse(self, stats: Union[List[List[float]], np.ndarray]) -> bool:
        self.count += 1
        self.stat = np.mean(stats, axis=0)
        ans = np.any(self.thresholds < self.stat - MeanFuseForWindowAlgo.EPS)

        # TODO: remove debug
        if ans:
            win_sizes = [size for size, threshold, stat in zip([200, 400, 800, 1600], self.thresholds, self.stat) if stat > threshold]
            print(f"detected by sizes {win_sizes}")
            if len(win_sizes) == 1:
                print("wow")
        return ans


class MedianFuseForWindowAlgo(Fuse):
    EPS = 1e-9

    def __init__(self, p: float, n_nodes: int, n_iter: int, window_sizes: List[Tuple[int, int]]):
        super(MedianFuseForWindowAlgo, self).__init__(p, n_nodes)
        self.count = 0
        rnd = np.random.RandomState(0)
        vec = np.load(os.path.join(
            os.path.dirname(__file__),
            f'../precalc_qunatiles/precalced_quantiles_{n_iter}_iter_with_burn-in.npy'
        ))[[[200, 400, 800, 1600].index(s) for s, _ in window_sizes]]
        vecs = [
            [rnd.permutation(sub_vec) for sub_vec in vec] for _ in range(n_nodes)
        ]
        fused_vec = np.median(vecs, axis=0)
        self.thresholds = np.quantile(fused_vec, 1 - p, axis=-1)

    def fuse(self, stats: Union[List[List[float]], np.ndarray]) -> bool:
        self.count += 1
        self.stat = np.median(stats, axis=0)
        ans = np.any(self.thresholds < self.stat - MedianFuseForWindowAlgo.EPS)

        # TODO: remove debug
        if ans:
            win_sizes = [size for size, threshold, stat in zip([200, 400, 800, 1600], self.thresholds, self.stat) if stat > threshold]
            print(f"detected by sizes {win_sizes}")
            if len(win_sizes) == 1:
                print("wow")
        return ans
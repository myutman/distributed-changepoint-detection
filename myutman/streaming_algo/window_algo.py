import os
from collections import deque
from typing import Tuple, Optional, List, Type, Union
import abc

import numpy as np

from myutman.streaming_algo.streaming_algo import StreamingAlgo


class TreeNode():
    EPS = 1e-7
    def __init__(self, key, val, prior):
        self.key = key
        self.val = val
        self.prior = prior
        self.max = max(0, val)
        self.min = min(0, val)
        self.sum = val
        self.left = None
        self.right = None

    def refresh(self):
        left_sum = 0 if self.left is None else self.left.sum
        self.max = left_sum + max(0, self.val)
        self.min = left_sum + min(0, self.val)
        self.sum = self.val
        if self.left is not None:
            self.max = max(self.max, self.left.max)
            self.min = min(self.min, self.left.min)
            self.sum = self.sum + self.left.sum
        if self.right is not None:
            self.max = max(self.max, self.right.max + left_sum + self.val)
            self.min = min(self.min, self.right.min + left_sum + self.val)
            self.sum = self.sum + self.right.sum


def split(node: Optional[TreeNode], key: float) -> Tuple[Optional[TreeNode], Optional[TreeNode]]:
    if node is None:
        return None, None
    if node.key < key - TreeNode.EPS:
        l, r = split(node.right, key)
        node.right = l
        node.refresh()
        return node, r
    l, r = split(node.left, key)
    node.left = r
    node.refresh()
    return l, node


def merge(node_l: Optional[TreeNode], node_r: Optional[TreeNode]) -> Optional[TreeNode]:
    if node_l is None:
        return node_r
    if node_r is None:
        return node_l
    if node_l.prior < node_r.prior:
        node_l.right = merge(node_l.right, node_r)
        node_l.refresh()
        return node_l
    node_r.left = merge(node_l, node_r.left)
    node_r.refresh()
    return node_r


class WindowTest:
    def __init__(self, reference: Union[List[float], np.ndarray], sliding: Union[List[float], np.ndarray]):
        self.reference = deque(reference)
        self.sliding = deque(sliding)

    @abc.abstractmethod
    def add_point(self, point: float):
        pass

    @abc.abstractmethod
    def get_stat(self) -> Tuple[float, float]:
        pass


class TreapWindowTest(WindowTest):
    def __init__(self, reference: Union[List[float], np.ndarray], sliding: Union[List[float], np.ndarray]):#, dist: Distance = KolmogorovSmirnovDistance()):
        super(TreapWindowTest, self).__init__(reference, sliding)
        self.rnd = np.random.RandomState(0)
        self.max = 0
        self.stat = 0
        self.grace = 0
        self.root = None

        for key in reference:
            self.insert(key, -1 / len(reference))
        for key in sliding:
            self.insert(key, 1 / len(sliding))

    def insert(self, key, value):
        l, r = split(self.root, key)
        l = merge(l, TreeNode(key, value, self.rnd.randint(low=0, high=2**30 - 1, dtype=np.uint32)))
        self.root = merge(l, r)

    def erase(self, key):
        l, r = split(self.root, key)
        l1, r2 = split(r, key + 2 * TreeNode.EPS)
        self.root = merge(l, r2)

    def add_point(self, point: float):
        key = self.sliding.popleft()
        self.sliding.append(point)

        self.erase(key)
        self.insert(point, 1 / len(self.sliding))

        self.stat = 0 if self.root is None else max(abs(self.root.min), abs(self.root.max))
        self.max = max(self.max, self.stat)

    def get_stat(self):
        return self.max


class WindowStreamingAlgo(StreamingAlgo):
    def __init__(
        self,
        p: float,
        n_iter: int,
        window_sizes: Optional[List[Tuple[int, int]]] = None,
        random_state: int = 0,
        window_test_type: Type[WindowTest] = TreapWindowTest
    ):
        super().__init__(p, random_state)
        if window_sizes is None:
            window_sizes = [(200, 200), (400, 400), (800, 800), (1600, 1600)]
        self.window_sizes: List[Tuple[int, int]] = window_sizes
        self.rnd = np.random.RandomState(random_state)
        self.window_test_type = window_test_type
        self.grace = 0

        self.vec = np.load(os.path.join(
            os.path.dirname(__file__),
            f'../precalc_qunatiles/precalced_quantiles_{n_iter}_iter_tmp.npy'
        ))[[[200, 400, 800, 1600].index(s) for s, _ in window_sizes]]

        # print(self.vec.shape)

        for i, _ in enumerate(window_sizes):
            self.vec[i, :] = self.rnd.permutation(self.vec[i, :])

        self.buffer = []
        self.window_tests: List[Optional[WindowTest]] = [
            None for _ in self.window_sizes
        ]

    def process_element(self, element, meta=None):
        enough = True
        for size_reference, size_sliding in self.window_sizes:
            if self.grace < size_reference + size_sliding:
                enough = False
        if not enough:
            self.buffer.append(element)
        self.grace += 1
        for i, (size_reference, size_sliding) in enumerate(self.window_sizes):
            if self.grace == size_reference + size_sliding:
                print(f'ref_size={size_reference}, slide_size={size_sliding}')
                reference = self.buffer[:size_reference]
                sliding = self.buffer[-size_sliding:]
                self.window_tests[i] = self.window_test_type(reference, sliding)
            elif self.grace > size_reference + size_sliding:
                self.window_tests[i].add_point(element)

    def get_stat(self):
        #tmp = np.array([
        #    [0 if window_test is None else window_test.get_stat()] for window_test in self.window_tests
        #])
        #stat = np.concatenate([tmp, self.vec], axis=-1)
        return np.array([0 if window_test is None else window_test.get_stat() for window_test in self.window_tests])

    """def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds"""

    """def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])"""

    def restart(self):
        self.grace = 0
        self.buffer.clear()
        self.window_tests = [
            None for _ in self.window_sizes
        ]

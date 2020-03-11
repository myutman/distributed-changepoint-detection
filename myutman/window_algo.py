from collections import deque
from typing import Tuple, Optional, List

import numpy as np

from myutman.distance import KolmogorovSmirnovDistance, Distance
from myutman.single_thread import StreamingAlgo


class TreeNode():
    EPS = 1e-7
    def __init__(self, key, val, prior):
        self.key = key
        self.val = val
        self.prior = prior
        self.max = val
        self.min = val
        self.sum = val
        self.left = None
        self.right = None

    def refresh(self):
        left_sum = 0 if self.left is None else self.left.sum
        self.max = left_sum + self.val
        self.min = left_sum + self.val
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


class WindowPair:
    def __init__(self, sizes):#, dist: Distance = KolmogorovSmirnovDistance()):
        self.sizes = sizes
        #self.__dist = dist
        self.rnd = np.random.RandomState(0)
        self.reference = deque()
        self.sliding = deque()
        self.max = 0
        self.stat = 0
        self.grace = 0
        self.root = None

    def insert(self, key, value):
        l, r = split(self.root, key)
        l = merge(l, TreeNode(key, value, self.rnd.randint(low=0, high=2**30 - 1, dtype=np.uint32)))
        self.root = merge(l, r)

    def erase(self, key):
        l, r = split(self.root, key)
        l1, r2 = split(r, key + 2 * TreeNode.EPS)
        self.root = merge(l, r2)

    def add_point(self, point):
        self.grace += 1
        if len(self.reference) < self.sizes[0]:
            self.reference.append(point)
            self.insert(point, - 1 / self.sizes[0])
        else:
            if len(self.sliding) == self.sizes[1]:
                key = self.sliding.popleft()
                self.erase(key)
            self.sliding.append(point)
            self.insert(point, 1 / self.sizes[1])
        if self.grace >= self.sizes[0] + self.sizes[1]:
            self.stat = 0 if self.root is None else max(abs(self.root.min), abs(self.root.max))
            self.max = max(self.max, self.stat)

    def get_stat(self):
        return self.max

    def clear(self):
        self.root = None
        self.reference.clear()
        self.sliding.clear()
        self.max = 0
        self.stat = 0
        self.grace = 0


class WindowStreamingAlgo(StreamingAlgo):
    def __init__(
        self,
        p: float,
        n_iter: int,
        window_sizes: Optional[List[Tuple[int, int]]] = None,
        random_state: int = 0
    ):
        super().__init__(p, random_state)
        if window_sizes is None:
            window_sizes = [(200, 200), (400, 400), (800, 800), (1600, 1600)]
        self.window_count = len(window_sizes)
        self.rnd = np.random.RandomState(random_state)

        # TODO: remove after experiments
        self.vec = np.load(f'precalced_quantiles_{n_iter}_iter.npy')
        for i in range(self.window_count):
            self.vec[i, :] = self.rnd.permutation(self.vec[i, :])
        # print(f'quantile inside window algo: {np.quantile(self.vec[0], 0.99)}')
        self.window_pairs = [
            WindowPair(sizes) for sizes in window_sizes
        ]

    def process_element(self, element, meta=None):
        for k in range(self.window_count):
            self.window_pairs[k].add_point(element)
            #for j in range(1, self.l + 1):
            #    self.window_pairs[k][j].add_point(self.rnd.uniform(0, 1))

    def get_stat(self):
        #tmp = np.array([
        #    [self.window_pairs[i][j].get_stat() for j in range(self.l + 1)] for i in range(self.window_count)
        #])
        #return tmp
        tmp = np.array([
            [self.window_pairs[i].get_stat()] for i in range(self.window_count)
        ])
        stat = np.concatenate([tmp, self.vec], axis=-1)
        #print(f'quantile inside window algo get_stat: {np.quantile(stat[0][1:], 0.99)}')
        return stat

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])

    def restart(self):
        #for list_pair in self.window_pairs:
        #    for pair in list_pair:
        #        pair.clear()
        for pair in self.window_pairs:
            pair.clear()

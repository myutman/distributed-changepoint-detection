"""import numpy as np

from myutman.distance import KolmogorovSmirnovDistance, Distance
from myutman.single_thread import StreamingAlgo

class WindowPair:
    def __init__(self, sizes, l: int = 100, dist: Distance = KolmogorovSmirnovDistance()):
        self.sizes = sizes
        self.__dist__ = dist
        self.l = l
        self.reference = np.array([[] for _ in range(l + 1)])
        self.sliding = np.array([[] for _ in range(l + 1)])
        self.max = np.zeros(l + 1)
        self.stat = np.zeros(l + 1)
        self.grace = 0
        self.rnd = np.random.RandomState(0)

    def add_point(self, point):
        samples = self.rnd.uniform(0, 1, size=self.l)
        self.grace += 1
        if len(self.reference) < self.sizes[0]:
            self.reference = np.hstack([self.reference, np.concatenate([[point], samples]).reshape(-1, 1)])
        if len(self.sliding) == self.sizes[1]:
            self.sliding = self.sliding[:,1:]
        self.sliding = np.hstack([self.sliding, np.concatenate([[point], samples]).reshape(-1, 1)])
        if self.grace >= self.sizes[0] + self.sizes[1]:
            self.stat = self.__dist__(self.reference, self.sliding)
            self.max = np.maximum(self.max, self.stat)

    def get_stat(self):
        return self.max

    def clear(self):
        self.reference = np.array([[] for _ in range(self.l + 1)])
        self.sliding = np.array([[] for _ in range(self.l + 1)])
        self.max = np.zeros(self.l + 1)
        self.stat = np.zeros(self.l + 1)
        self.grace = 0

class WindowStreamingAlgo(StreamingAlgo):
    def __init__(self, p, l=None, window_sizes=None, dist: Distance = KolmogorovSmirnovDistance()):
        super().__init__(p)
        if l is None:
            l = 30
        self.l = l
        if window_sizes is None:
            #window_sizes = [(50 + self.rnd.choice(30), 50 + self.rnd.choice(30)) for _ in range(3)]
            window_sizes = [(20, 20), (30, 30), (40, 40)]
        self.window_count = len(window_sizes)
        self.window_pairs = [WindowPair(sizes, l, dist) for sizes in window_sizes]

    def process_element(self, element, meta=None):
        for k in range(self.window_count):
            self.window_pairs[k].add_point(element)

    def get_stat(self):
        tmp = np.array([self.window_pairs[i].get_stat() for i in range(self.window_count)])
        return tmp

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])

    def restart(self):
        for pair in self.window_pairs:
            pair.clear()
"""
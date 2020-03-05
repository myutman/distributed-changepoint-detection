import numpy as np

from myutman.distance import KolmogorovSmirnovDistance, Distance
from myutman.single_thread import StreamingAlgo


class WindowPair:
    def __init__(self, sizes, dist: Distance = KolmogorovSmirnovDistance()):
        self.sizes = sizes
        self.__dist = dist
        self.reference = []
        self.sliding = []
        self.max = 0
        self.stat = 0
        self.grace = 0

    def add_point(self, point):
        self.grace += 1
        if len(self.reference) < self.sizes[0]:
            self.reference.append(point)
        if len(self.sliding) == self.sizes[1]:
            self.sliding = self.sliding[1:]
        self.sliding.append(point)
        if self.grace >= self.sizes[0] + self.sizes[1]:
            self.stat = self.__dist(self.reference, self.sliding)
            self.max = max(self.max, self.stat)

    def get_stat(self):
        return self.max

    def clear(self):
        self.reference = []
        self.sliding = []
        self.max = 0
        self.stat = 0
        self.grace = 0


class WindowStreamingAlgo(StreamingAlgo):
    def __init__(self, p, l=None, window_sizes=None, dist: Distance = KolmogorovSmirnovDistance()):
        super().__init__(p)
        if l is None:
            l = 30
        self.l = l
        self.rnd = np.random.RandomState(0)
        if window_sizes is None:
            #window_sizes = [(50 + self.rnd.choice(30), 50 + self.rnd.choice(30)) for _ in range(3)]
            window_sizes = [(20, 20), (30, 30), (40, 40)]
        self.window_count = len(window_sizes)
        self.window_pairs = [[WindowPair(sizes, dist) for _ in range(l + 1)] for sizes in window_sizes]

    def process_element(self, element, meta=None):
        for k in range(self.window_count):
            self.window_pairs[k][0].add_point(element)
            for j in range(1, self.l + 1):
                self.window_pairs[k][j].add_point(self.rnd.uniform(0, 1))

    def get_stat(self):
        tmp = np.array([
            [self.window_pairs[i][j].get_stat() for j in range(self.l + 1)] for i in range(self.window_count)
        ])
        return tmp

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])

    def restart(self):
        for list_pair in self.window_pairs:
            for pair in list_pair:
                pair.clear()

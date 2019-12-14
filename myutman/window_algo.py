from myutman.single_thread import StreamingAlgo
from myutman.distributed import RoundrobinStreamingAlgo

import numpy as np


def kolmogorov_smirnov_dist(reference_window, sliding_window):
    lst = []
    for v in reference_window:
        lst.append((v, - 1 / len(reference_window)))
    for v in sliding_window:
        lst.append((v, 1 / len(sliding_window)))
    lst = sorted(lst)
    cur = 0
    mx = 0
    for v, p in lst:
        cur += p
        mx = max(mx, abs(cur))
    return mx


class WindowPair:

    def __init__(self, sizes, dist=kolmogorov_smirnov_dist):
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

    def __init__(self, p, l, window_sizes, dist=kolmogorov_smirnov_dist):
        super().__init__(p)
        self.l = l
        self.window_count = len(window_sizes)
        self.window_pairs = [[WindowPair(sizes, dist) for _ in range(l + 1)] for sizes in window_sizes]
        self.__dist = dist

    def process_element(self, element, meta=None):
        for k in range(self.window_count):
            self.window_pairs[k][0].add_point(element)
            for j in range(1, self.l + 1):
                self.window_pairs[k][j].add_point(np.random.uniform(0, 1))

    def get_stat(self):
        return np.array([[self.window_pairs[i][j].get_stat() for j in range(self.l + 1)] for i in range(self.window_count)])

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])

    def restart(self):
        for list_pair in self.window_pairs:
            for pair in list_pair:
                pair.clear()


class WindowRoundrobinStreamingAlgo(RoundrobinStreamingAlgo):
    def __init__(self, p, n_nodes, l, window_sizes, dist=kolmogorov_smirnov_dist):
        single_threads = [WindowStreamingAlgo(p, l, window_sizes, dist) for _ in range(n_nodes)]
        super(WindowRoundrobinStreamingAlgo, self).__init__(p, single_threads)

    def fuse(self, stats):
        ans = np.max(stats, axis=0)
        return ans

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])


class WindowDependentStreamingAlgo(RoundrobinStreamingAlgo):
    def __init__(self, p, n_nodes, l, window_sizes, dist=kolmogorov_smirnov_dist):
        single_threads = [WindowStreamingAlgo(p, l, window_sizes, dist) for _ in range(n_nodes)]
        super(WindowDependentStreamingAlgo, self).__init__(p, single_threads)

    def fuse(self, stats):
        ans = np.max(stats, axis=0)
        return ans

    def get_thresholds(self):
        thresholds = np.quantile(self.get_stat()[:,1:], 1 - self.p, axis=-1)
        return thresholds

    def test(self):
        return np.any(self.get_thresholds() < self.get_stat()[:,0])


if __name__ == '__main__':
    algo = WindowRoundrobinStreamingAlgo(0.05, 5, 30, [(71, 93), (80, 90), (65, 79)])
    array = list(np.random.normal(0, 1, size=1000))
    array += list(np.random.normal(0.2, 3, size=1000))

    #print(kolmogorov_smirnov_dist(np.random.normal(0, 1, size = 100), np.random.normal(100, 1, size = 100)))
    for it, elem in enumerate(array):
        algo.process_element(elem)
        print(algo.get_stat(), algo.get_thresholds())
        if algo.test():
            print(it)
            break
from myutman.exceptions import UnimplementedException
from myutman.exceptions import IncorrectDataException
import numpy as np

class StreamingAlgo:

    def __init__(self, p):
        self.p = p

    def process_element(self, element):
        raise UnimplementedException()

    def get_stat(self):
        raise UnimplementedException()

    def test(self):
        raise UnimplementedException()

def kholmogorov_smirnov_dist(reference_window, sliding_window):
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
        #print(p, cur, mx)
    return mx

class WindowPair:

    def __init__(self, sizes, dist=kholmogorov_smirnov_dist):
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

    def get_max(self):
        return self.max

    def get_stat(self):
        return self.stat


class WindowStreamingAlgo(StreamingAlgo):

    def __init__(self, p, l, window_sizes, dist=kholmogorov_smirnov_dist):
        super().__init__(p)
        print(self.p)
        self.l = l
        self.window_count = len(window_sizes)
        self.window_pairs = [[WindowPair(sizes, dist) for _ in range(l + 1)] for sizes in window_sizes]
        self.__dist = dist

    def process_element(self, element):
        #self.grace += 1
        for k in range(self.window_count):
            self.window_pairs[k][0].add_point(element)
            for j in range(1, self.l + 1):
                self.window_pairs[k][j].add_point(np.random.uniform(0, 1))

    def get_stat(self):
        return [self.window_pairs[i][0].get_stat() for i in range(self.window_count)]

    def get_thresholds(self):
        thresholds = []
        for k in range(self.window_count):
            maxes = [self.window_pairs[k][i].get_max() for i in range(1, self.l + 1)]
            thresholds.append(np.quantile(maxes, 1 - self.p))
        return thresholds

    def test(self):
        return np.any(list(map(lambda x: x[0] < x[1], zip(self.get_thresholds(), self.get_stat()))))


if __name__ == '__main__':
    algo = WindowStreamingAlgo(0.05, 30, [(71, 93), (80, 90), (65, 79)])
    array = list(np.random.normal(0, 1, size = 1000))
    array += list(np.random.normal(0.2, 3, size = 1000))

    #print(kholmogorov_smirnov_dist(np.random.normal(0, 1, size = 100), np.random.normal(100, 1, size = 100)))
    for it, elem in enumerate(array):
        algo.process_element(elem)
        print(algo.get_stat(), algo.get_thresholds())
        if algo.test():
            print(it)
            break

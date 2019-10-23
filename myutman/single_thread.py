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

class WindowStreamingAlgo(StreamingAlgo):

    def __init__(self, p, l, window_count, window_sizes, dist = kholmogorov_smirnov_dist):
        super().__init__(p)
        print(self.p)
        if window_count != len(window_sizes):
            raise IncorrectDataException()
        self.l = l
        self.window_count = window_count
        self.maxes = [[0 for _ in range(l)] for _ in range(window_count)]
        self.reference_windows = [[[] for _ in range(l + 1)] for _ in range(window_count)]
        self.sliding_windows = [[[] for _ in range(l + 1)] for _ in range(window_count)]
        self.window_sizes = window_sizes
        self.__dist = dist
        self.grace = 0

    def process_element(self, element):
        self.grace += 1
        for k in range(self.window_count):
            if len(self.reference_windows[k][0]) < self.window_sizes[k][0]:
                self.reference_windows[k][0].append(element)
            if len(self.sliding_windows[k][0]) == self.window_sizes[k][1]:
                self.sliding_windows[k][0] = self.sliding_windows[k][0][1:]
            self.sliding_windows[k][0].append(element)
            for j in range(1, self.l + 1):
                new_val = np.random.uniform(0, 1)
                if len(self.reference_windows[k][j]) < self.window_sizes[k][0]:
                    self.reference_windows[k][j].append(new_val)
                if len(self.sliding_windows[k][j]) == self.window_sizes[k][1]:
                    self.sliding_windows[k][j] = self.sliding_windows[k][j][1:]
                self.sliding_windows[k][j].append(new_val)
                if self.grace >= self.window_sizes[k][0] + self.window_sizes[k][1]:
                    self.maxes[k][j - 1] = max(self.maxes[k][j - 1],
                        self.__dist(self.reference_windows[k][j], self.sliding_windows[k][j]))

    def get_stat(self):
        return [0 if self.grace < self.window_sizes[i][0] + self.window_sizes[i][1]
                else self.__dist(self.reference_windows[i][0], self.sliding_windows[i][0])
                for i in range(self.window_count)]

    def get_threshholds(self):
        threshholds = []
        for k in range(self.window_count):
            threshholds.append(np.quantile(self.maxes[k], 1 - self.p))
        return threshholds

    def test(self):
        return np.any(list(map(lambda x: x[0] < x[1], zip(self.get_threshholds(), self.get_stat()))))


if __name__ == '__main__':
    algo = WindowStreamingAlgo(0.05, 30, 3, [(71, 93), (80, 90), (65, 79)])
    array = list(np.random.normal(0, 1, size = 1000))
    array += list(np.random.normal(0.2, 3, size = 1000))

    #print(kholmogorov_smirnov_dist(np.random.normal(0, 1, size = 100), np.random.normal(100, 1, size = 100)))
    for it, elem in enumerate(array):
        algo.process_element(elem)
        print(algo.get_stat(), algo.get_threshholds())
        if algo.test():
            print(it)
            break

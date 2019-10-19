from myutman.exceptions import UnimplementedException
from myutman.exceptions import IncorrectDataException
import numpy as np

class StreamingAlgo:

    def __init__(self):
        pass

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

    def __init__(self, window_count, window_sizes, threshholds, dist = kholmogorov_smirnov_dist):
        super().__init__()
        if window_count != len(window_sizes) or window_count != len(threshholds):
            raise IncorrectDataException()
        self.window_count = window_count
        self.reference_windows = [[] for i in range(window_count)]
        self.sliding_windows = [[] for i in range(window_count)]
        self.window_sizes = window_sizes
        self.threshholds = threshholds
        self.__dist = dist

    def process_element(self, element):
        for k in range(self.window_count):
            if len(self.reference_windows[k]) < self.window_sizes[k][0]:
                self.reference_windows[k].append(element)
            if len(self.sliding_windows[k]) == self.window_sizes[k][1]:
                self.sliding_windows[k] = self.sliding_windows[k][1:]
            self.sliding_windows[k].append(element)

    def get_stat(self):
        return [0 if (len(self.reference_windows[i]) < self.window_sizes[i][0]) or (len(self.sliding_windows[i]) < self.window_sizes[i][1]) else self.__dist(self.reference_windows[i], self.sliding_windows[i]) for i in range(self.window_count)]

    def test(self):
        return np.any(list(map(lambda x: x[0] < x[1], zip(self.threshholds, self.get_stat()))))


if __name__ == '__main__':
    algo = WindowStreamingAlgo(3, [(71, 93), (80, 90), (65, 79)], [0.4, 0.4, 0.4])
    array = list(np.random.normal(0, 1, size = 1000))
    array += list(np.random.normal(0.2, 3, size = 1000))

    #print(kholmogorov_smirnov_dist(np.random.normal(0, 1, size = 100), np.random.normal(100, 1, size = 100)))
    for it, elem in enumerate(array):
        algo.process_element(elem)
        print(algo.get_stat())
        if algo.test():
            print(it)
            break

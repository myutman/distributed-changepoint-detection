import abc
from typing import List


class Distance:

    @abc.abstractmethod
    def get_dist(self, reference_window: List[float], sliding_window: List[float]) -> float:
        pass

    def __call__(self, reference_window: List[float], sliding_window: List[float]) -> float:
        return self.get_dist(reference_window, sliding_window)


class KolmogorovSmirnovDistance(Distance):
    def __init__(self):
        super(KolmogorovSmirnovDistance, self).__init__()

    def get_dist(self, reference_window: List[float], sliding_window: List[float]) -> float:
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

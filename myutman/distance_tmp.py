"""import abc
import numpy as np
import sortednp as snp
from typing import List, Union


class Distance:

    @abc.abstractmethod
    def get_dist(self, reference_window: np.ndarray, sliding_window: np.ndarray) -> Union[np.ndarray, float]:
        pass

    def __call__(self, reference_window: np.ndarray, sliding_window: np.ndarray) -> Union[np.ndarray, float]:
        return self.get_dist(reference_window, sliding_window)


class KolmogorovSmirnovDistance(Distance):
    def __init__(self):
        super(KolmogorovSmirnovDistance, self).__init__()

    def get_dist(self, reference_window: np.ndarray, sliding_window: np.ndarray) -> Union[np.ndarray, float]:
        assert len(reference_window.shape) == len(sliding_window.shape)
        assert reference_window.shape[:-1] == sliding_window.shape[:-1]
        #reference_window = reference_window.copy()
        #sliding_window = sliding_window.copy()

        if len(reference_window.shape) == 1:
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

        vals = np.concatenate([-np.ones(reference_window.shape[-1]) / reference_window.shape[-1], np.ones(sliding_window.shape[-1]) / sliding_window.shape[-1]])
        union = np.concatenate([reference_window, sliding_window], axis=-1)
        indices = union.argsort(axis=-1)
        window_vals = vals[indices]
        return np.abs(window_vals.cumsum(axis=-1)).max(axis=-1)
"""
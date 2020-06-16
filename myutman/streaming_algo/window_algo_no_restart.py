import os
from typing import Optional, List, Tuple, Type

import numpy as np

from myutman.streaming_algo.streaming_algo import StreamingAlgo
from myutman.streaming_algo.window_algo import WindowTest, TreapWindowTest


class WindowStreamingAlgoNoRestart(StreamingAlgo):
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
        return np.array([0 if window_test is None else window_test.get_stat()[1] for window_test in self.window_tests])

    def restart(self):
        pass
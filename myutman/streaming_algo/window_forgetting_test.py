from typing import List, Union

import numpy as np

from myutman.streaming_algo.window_algo import TreapWindowTest


class ForgettingTreapWindowTest(TreapWindowTest):
    def __init__(self, reference: Union[List[float], np.ndarray], sliding: Union[List[float], np.ndarray], forgetting_weight: float = 0.9):#, dist: Distance = KolmogorovSmirnovDistance()):
        super(ForgettingTreapWindowTest, self).__init__(reference, sliding)
        self.forgetting_stat = 0
        self.forgetting_weight_sum = 0
        self.forgetting_weight = forgetting_weight

    def add_point(self, point: float):
        super(ForgettingTreapWindowTest, self).add_point(point)
        self.forgetting_stat = self.forgetting_stat * self.forgetting_weight + self.stat
        self.forgetting_weight_sum = self.forgetting_weight_sum * self.forgetting_weight + 1

    def get_stat(self):
        return 0 if self.forgetting_weight_sum == 0 else self.forgetting_stat / self.forgetting_weight_sum

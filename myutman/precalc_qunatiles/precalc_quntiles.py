from collections import defaultdict

import numpy as np
from tqdm import tqdm

from myutman.streaming_algo.window_algo import TreapWindowTest

if __name__ == '__main__':
    sizes = [(200, 200), (400, 400), (800, 800), (1600, 1600)]
    niters_to_save = {
        1250, 2500, 5000, 20000,
        3125, 6250, 12500, 50000
    }

    rnd = np.random.RandomState(0)

    """

    
    window_pairs = [[TreapWindowTest(
        reference=rnd.uniform(0, 1, size=size1),
        sliding=rnd.uniform(0, 1, size=size1)
    ) for j in range(500)] for size1, size2 in sizes]"""

    """for iter in tqdm(range(1, 50001)):
        for i, window_pairs_by_size in enumerate(window_pairs):
            for j, pair in enumerate(window_pairs_by_size):
                pair.add_point(rnd.uniform(0, 1))
                if iter in niters_to_save:
                    vec_by_niters[iter + sizes[i][0] + sizes[i][1]][i][j] = pair.get_stat()
        if iter in niters_to_save:
            np.save(f'precalced_quantiles_{iter}_iter_with_burn-in.npy', vec_by_niters[iter])"""

    vec_by_niters = defaultdict(lambda: [[] for _ in sizes])

    for i in tqdm(range(500)):
        for j, (size1, size2) in enumerate(sizes):
            window_pair = TreapWindowTest(
                reference=rnd.uniform(0, 1, size=size1),
                sliding=rnd.uniform(0, 1, size=size2)
            )
            for iter in range(1, 20001):
                window_pair.add_point(rnd.uniform(0, 1))
                if iter in niters_to_save:
                    vec_by_niters[iter][j].append(window_pair.get_stat())
        for iter in niters_to_save:
            np.save(f'precalced_quantiles_{iter}_iter_with_burn-in.npy', vec_by_niters[iter])
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from myutman.streaming_algo.window_algo import TreapWindowTest

if __name__ == '__main__':
    sizes = [(200, 200), (400, 400), (800, 800), (1600, 1600)]

    vec_by_niters = defaultdict(lambda: [np.zeros(500) for _ in sizes])

    rnd = np.random.RandomState(0)

    vecs = [[] for _ in sizes]
    for i in tqdm(range(500)):
        for j, (size1, size2) in enumerate(sizes):
            window_pairs = [TreapWindowTest(
                reference=rnd.uniform(0, 1, size=size1),
                sliding=rnd.uniform(0, 1, size=size2)
            ) for _ in range(4)]
            for iter in range(1, 20001 - 4 * (size1 - size2)):
                window_pairs[iter % 4].add_point(rnd.uniform(0, 1))
            vecs[j].append(np.max([pair.get_stat() for pair in window_pairs]))
        np.save('precalced_quantiles_20000_iter_4_nodes.npy', vecs)

import numpy as np
from tqdm import tqdm

from myutman.streaming_algo.window_algo import WindowPair

if __name__ == '__main__':
    sizes = [(200, 200), (400, 400), (800, 800), (1600, 1600)]
    n_iters_to_save = {
        1250, 2500, 5000, 20000,
        3125, 6250, 12500, 50000
    }
    window_pairs = [[WindowPair(sizes=size) for j in range(100)] for size in sizes]
    rnd = np.random.RandomState(0)
    for iter in tqdm(range(1, 50001)):
        vec = [[] for _ in sizes]
        for i, window_pairs_by_size in enumerate(window_pairs):
            for pair in window_pairs_by_size:
                pair.add_point(rnd.uniform(0, 1))
                vec[i].append(pair.get_stat())
        if iter in n_iters_to_save:
            np.save(f'precalced_quantiles_{iter}_iter_tmp.npy', vec)



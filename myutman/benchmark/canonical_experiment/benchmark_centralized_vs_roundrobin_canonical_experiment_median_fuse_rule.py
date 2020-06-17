import json

from myutman.fuse.fuse import MedianFuseForWindowAlgo
from myutman.generation.generation import SimpleMultichangeSampleGeneration
from myutman.node_distribution.node_distribution import RoundrobinNodeDistribution
from myutman.stand.stand import Stand
from myutman.streaming_algo.window_algo import WindowStreamingAlgo

if __name__ == '__main__':
    p_levels = [0.01, 0.05, 0.1]
    n_nodess = [4, 8, 16]

    windows = [(200, 200), (400, 400), (800, 800), (1600, 1600)]

    n_iter = 20000

    stand_centralized = Stand(
        n_nodes=1,
        algo=WindowStreamingAlgo,
        account1_node_distribution=RoundrobinNodeDistribution,
        account2_node_distribution=RoundrobinNodeDistribution,
        fuse=MedianFuseForWindowAlgo,
        account1_algo_kwargs={"window_sizes": windows, "n_iter": n_iter},
        account2_algo_kwargs={"window_sizes": windows, "n_iter": n_iter}
    )
    stand_roundrobins = [
        Stand(
            n_nodes=n_nodes,
            algo=WindowStreamingAlgo,
            account1_node_distribution=RoundrobinNodeDistribution,
            account2_node_distribution=RoundrobinNodeDistribution,
            fuse=MedianFuseForWindowAlgo,
            account1_algo_kwargs={"window_sizes": windows, "n_iter": n_iter // n_nodes},
            account2_algo_kwargs={"window_sizes": windows, "n_iter": n_iter // n_nodes},
        ) for n_nodes in n_nodess
    ]

    stands = [
        stand_centralized,
        *stand_roundrobins
    ]

    generations = [
        SimpleMultichangeSampleGeneration
    ]

    results = [[[[] for _ in p_levels] for _ in stands] for _ in generations]
    for state in range(5):
        for i, generation in enumerate(generations):
            sample, change_points, change_points_ids = generation(state=state)(
                size=400000,
                change_period=20000,
                change_period_noise=1,
                n_modes=1
            )
            for j, stand in enumerate(stands):
                for k, p_level in enumerate(p_levels):
                    result1 = stand.test(
                        p=p_level,
                        sample=sample,
                        change_points=change_points,
                        change_ids=change_points_ids,
                        n_account1s=1,
                        n_account2s=0
                    )
                    print(result1)
                    results[i][j][k].append(result1)
        with open(f'../../results/centralized_vs_roundrobin_p={p_levels}_nnodes={n_nodess}_canonical_experiment_median_fuse_rule.json', 'w') as output_file:
            json.dump(results, output_file, indent=4, ensure_ascii=False)
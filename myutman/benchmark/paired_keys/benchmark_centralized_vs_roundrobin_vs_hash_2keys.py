from importlib import reload

import myutman
reload(myutman.stand.stand_utils)
reload(myutman.stand.stand)
reload(myutman.streaming_algo.window_algo)
reload(myutman.generation.generation)
reload(myutman.fuse.fuse)

from myutman.fuse.fuse import FuseForWindowAlgo
from myutman.generation.generation import ChangeSampleGeneration
from myutman.node_distribution.node_distribution import RoundrobinNodeDistribution, DependentNodeDistribution, \
    SecondMetaDependentNodeDistribution
from myutman.stand import Stand
from myutman.streaming_algo.window_algo import WindowStreamingAlgo

import json


if __name__ == '__main__':
    p_levels = [0.01, 0.05, 0.1]
    n_nodess = [4, 8, 16]
    
    big_windows = [(100, 100), (150, 150), (200, 200)]
    small_windows = lambda n: [(a // n, b // n) for a, b in big_windows]

    n_account1s = 5
    n_account2s = 5

    stand_centralized = Stand(
        n_nodes=1,
        algo=WindowStreamingAlgo,
        account1_node_distribution=RoundrobinNodeDistribution,
        account2_node_distribution=RoundrobinNodeDistribution,
        fuse=FuseForWindowAlgo,
        account1_algo_kwargs={"window_sizes": big_windows},
        account2_algo_kwargs={"window_sizes": big_windows}
    )
    stand_roundrobins = [
        Stand(
            n_nodes=n_nodes,
            algo=WindowStreamingAlgo,
            account1_node_distribution=RoundrobinNodeDistribution,
            account2_node_distribution=RoundrobinNodeDistribution,
            fuse=FuseForWindowAlgo,
            account1_algo_kwargs={"window_sizes": small_windows(n_nodes)},
            account2_algo_kwargs={"window_sizes": small_windows(n_nodes)}
        ) for n_nodes in n_nodess
    ]
    stand_account1_dependents = [
        Stand(
            n_nodes=n_nodes,
            algo=WindowStreamingAlgo,
            account1_node_distribution=DependentNodeDistribution,
            account2_node_distribution=SecondMetaDependentNodeDistribution,
            fuse=FuseForWindowAlgo,
            account1_algo_kwargs={"window_sizes": big_windows},
            account2_algo_kwargs={"window_sizes": small_windows(n_nodes)}
        ) for n_nodes in n_nodess
    ]
    stand_account2_dependents = [
        Stand(
            n_nodes=n_nodes,
            algo=WindowStreamingAlgo,
            account1_node_distribution=SecondMetaDependentNodeDistribution,
            account2_node_distribution=DependentNodeDistribution,
            fuse=FuseForWindowAlgo,
            account1_algo_kwargs={"window_sizes": small_windows(n_nodes)},
            account2_algo_kwargs={"window_sizes": big_windows}
        ) for n_nodes in n_nodess
    ]

    stands = [
        stand_centralized,
        *stand_roundrobins,
        *stand_account1_dependents,
        *stand_account2_dependents
    ]

    generations = [
        ChangeSampleGeneration
    ]

    results = [[[[] for _ in p_levels] for _ in stands] for _ in generations]
    for state in range(100):
        for i, generation in enumerate(generations):
            sample, change_points, change_points_ids = generation(state=state)(
                size=101000,
                n_clients=n_account1s,
                n_terminals=n_account2s,
                change_period=1000,
                change_period_noise=0,
                change_interval=1000,
                change_amount=10,
                change_amount_noise=0
            )
            for j, stand in enumerate(stands):
                for k, p_level in enumerate(p_levels):
                    result1 = stand.test(
                        p=p_level,
                        sample=sample,
                        change_points=change_points,
                        change_ids=change_points_ids,
                        n_account1s=n_account1s,
                        n_account2s=n_account2s
                    )
                    print(result1)
                    results[i][j][k].append(result1)
        with open(f'centralized_vs_roundrobin_p={p_levels}_nnodes={n_nodess}.json') as output_file:
            json.dump(results, output_file, indent=4, ensure_ascii=False)
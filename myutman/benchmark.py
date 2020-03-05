from importlib import reload

import myutman

reload(myutman.stand_utils)
reload(myutman.stand)
reload(myutman.window_algo)
reload(myutman.generation)
reload(myutman.fuse)

from myutman.fuse import FuseForWindowAlgo
from myutman.generation import ChangeSampleGeneration
from myutman.node_distribution import RoundrobinNodeDistribution, DependentNodeDistribution, \
    SecondMetaDependentNodeDistribution
from myutman.stand import Stand
from myutman.window_algo import WindowStreamingAlgo

if __name__ == '__main__':
    n_nodes = 5
    stand_centralized = Stand(
        n_nodes=1,
        algo=WindowStreamingAlgo,
        client_node_distribution=RoundrobinNodeDistribution,
        terminal_node_distribution=RoundrobinNodeDistribution,
        fuse=FuseForWindowAlgo(),
        client_algo_kwargs={"window_sizes": [(20, 20), (30, 30), (40, 40)]},
        terminal_algo_kwargs={"window_sizes": [(20, 20), (30, 30), (40, 40)]}
    )
    stand_roundrobins = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=RoundrobinNodeDistribution,
        terminal_node_distribution=RoundrobinNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    stand_client_dependent = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=DependentNodeDistribution,
        terminal_node_distribution=SecondMetaDependentNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    stand_terminal_dependent = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=SecondMetaDependentNodeDistribution,
        terminal_node_distribution=DependentNodeDistribution,
        fuse=FuseForWindowAlgo()
    )

    stands = [
        stand_centralized,
        #stand_roundrobins,
        #stand_client_dependent,
        #stand_terminal_dependent
    ]

    generations = [
        ChangeSampleGeneration
    ]

    n_clients = 5
    n_terminals = 5

    results = [[[] for _ in stands] for _ in generations]
    for state in range(1):
        for i, generation in enumerate(generations):
            sample, change_points, change_points_ids = generation(state=state)(
                size=20000,
                n_clients=n_clients,
                n_terminals=n_terminals,
                change_period=400,
                change_period_noise=0,
                change_interval=400
            )
            for j, stand in enumerate(stands):
                result1 = stand.test(
                    p=0.05,
                    sample=sample,
                    change_points=change_points,
                    change_ids=change_points_ids,
                    n_clients=n_clients,
                    n_terminals=n_terminals
                )
                print(result1)
                results[i][j].append(result1)
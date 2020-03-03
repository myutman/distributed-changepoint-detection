from myutman.fuse import FuseForWindowAlgo
from myutman.node_distribution import RoundrobinNodeDistribution, DependentNodeDistribution, \
    SecondMetaDependentNodeDistribution
from myutman.stand import Stand
from myutman.generation import ClientTerminalsReorderSampleGeneration
from myutman.stand_utils import compare_results
from myutman.window_algo import WindowStreamingAlgo

if __name__ == '__main__':
    #stand = Stand([WindowStreamingAlgo], [FuseForWindowAlgo()], [RoundrobinNodeDistribution, DependentNodeDistribution], [ClientTerminalsReorderSampleGeneration])
    #stand.compare_distibuted_algorithms_plots()

    n_nodes = 10
    stand = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=RoundrobinNodeDistribution,
        terminal_node_distribution=RoundrobinNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    stand1 = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=DependentNodeDistribution,
        terminal_node_distribution=SecondMetaDependentNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    stand2 = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=SecondMetaDependentNodeDistribution,
        terminal_node_distribution=DependentNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    stand3 = Stand(
        n_nodes=n_nodes,
        algo=WindowStreamingAlgo,
        client_node_distribution=DependentNodeDistribution,
        terminal_node_distribution=DependentNodeDistribution,
        fuse=FuseForWindowAlgo()
    )
    results1 = []
    results2 = []
    for i in range(10):
        sample, change_points = ClientTerminalsReorderSampleGeneration(state=i)(size=10000, change_period=100, change_interval=20)
        result1 = stand.test(
            p=0.05,
            sample=sample,
            change_points=change_points,
            n_clients=20,
            n_terminals=20
        )
        print(result1)
        results1.append(result1)
        result2 = stand1.test(
            p=0.05,
            sample=sample,
            change_points=change_points,
            n_clients=20,
            n_terminals=20
        )
        print(result2)
        results2.append(result2)
    print(compare_results(results1, results2))


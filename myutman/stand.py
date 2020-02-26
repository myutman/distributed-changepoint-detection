from typing import Tuple, List, Dict, Type, Any

from myutman.fuse import Fuse
from myutman.generation import SampleGeneration
from myutman.node_distribution import NodeDistribution
from myutman.single_thread import StreamingAlgo

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from myutman.stand_utils import calc_error


class Stand:
    def __init__(
        self,
        n_nodes,
        algo: Type[StreamingAlgo],
        client_node_distribution: Type[NodeDistribution],
        terminal_node_distribution: Type[NodeDistribution],
        fuse: Fuse
    ) -> None:
        self.n_nodes = n_nodes
        self.algo = algo
        self.client_node_distribution = client_node_distribution(n_nodes)
        self.terminal_node_distribution = terminal_node_distribution(n_nodes)
        self.fuse = fuse

    def test(
        self,
        p: float,
        sample: List[Tuple[Any, float]],
        change_points: List[int],
        n_clients,
        n_terminals
    ) -> Dict[str, float]:
        nodes: np.ndarray = np.array([#List[Tuple[List[StreamingAlgo], List[StreamingAlgo]]] = np.array([
            ([self.algo(p) for  _ in range(n_clients)],
             [self.algo(p) for  _ in range(n_terminals)
              ]) for _ in range(self.n_nodes)
        ])
        detected = []
        for i, ((client_id, terminal_id), point) in tqdm(enumerate(sample)):
            client_node_id = self.client_node_distribution.get_node_index(client_id, terminal_id)
            terminal_node_id = self.terminal_node_distribution.get_node_index(terminal_id, client_id)
            nodes[client_node_id][0][client_id].process_element(point)
            nodes[terminal_node_id][1][terminal_id].process_element(point)
            detection = False
            if self.fuse(p, nodes[:, 1, terminal_id]):
                detection = True
                for node in nodes[:, 1, terminal_id]:
                    node.restart()
            if self.fuse(p, nodes[:, 0, client_id]):
                detection = True
                for node in nodes[:, 0, client_id]:
                    node.restart()
            if detection:
                detected.append(i)

        return calc_error(change_points, detected)


"""
class Stand:
    def __init__(
        self,
        algos: List[Type[StreamingAlgo]],
        fuses: List[Fuse],
        client_node_distributions: List[Type[NodeDistribution]],

        sample_generations: List[Type[SampleGeneration]]
    ):
        self.algos = algos
        self.fuses = fuses
        self.node_distributions = node_distributions
        self.sample_generations = sample_generations

    def __run_test__(
            self,
            p: float,
            algo: Type[StreamingAlgo],
            fuse: Fuse,
            node_distribution_type: Type[NodeDistribution],
            n_nodes: int,
            generate_sample: SampleGeneration
    ) -> Dict[str, float]:
        sample, change_points = generate_sample(size=100000)
        nodes: List[StreamingAlgo] = [algo(p) for _ in range(n_nodes)]
        detected = []
        node_distribution = node_distribution_type(n_nodes)
        for i, (meta, point) in tqdm(enumerate(sample)):
            node_id = node_distribution.get_node_index(meta)
            nodes[node_id].process_element(point, meta)
            if fuse(p, nodes):
                detected.append(i)
                for node in nodes:
                    node.restart()

        return calc_error(change_points, detected)

    def compare_distibuted_algorithms_plots(self, p: float = 0.05, rng: List[int] = range(1, 3)):
        tdrs = dict([(f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions for sample_generation in self.sample_generations])
        mdrs = dict([(f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions for sample_generation in self.sample_generations])
        fdrs = dict([(f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions for sample_generation in self.sample_generations])



        for n_nodes in rng:
            for node_distribution in self.node_distributions:
                for algo, fuse in zip(self.algos, self.fuses):
                    for sample_generation in self.sample_generations:
                        result = self.__run_test__(p, algo, fuse, node_distribution, n_nodes, sample_generation())
                        print(f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}: {result}')
                        tdrs[f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}'].append(result['TDR'])
                        mdrs[f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}'].append(result['MDR'])
                        fdrs[f'{algo.__name__} {node_distribution.__name__} {sample_generation.__name__}'].append(result['FDR'])

        fig = plt.figure(figsize=(15, 15))
        ax = fig.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
        for lst in list(tdrs.values()):
            ax[0].plot(rng, lst)
        ax[0].legend(list(tdrs.keys()))
        ax[0].set_title('TDR')

        for lst in list(mdrs.values()):
            ax[1].plot(rng, lst)
        ax[1].legend(list(mdrs.keys()))
        ax[1].set_title('MDR')

        for lst in list(fdrs.values()):
            ax[2].plot(rng, lst)
        ax[2].legend(list(fdrs.keys()))
        ax[2].set_title('FDR')

        plt.show()
"""
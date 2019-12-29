from typing import Tuple, List, Dict, Type

from myutman.fuse import Fuse
from myutman.node_distribution import NodeDistribution
from myutman.single_thread import StreamingAlgo

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from myutman.stand_utils import calc_error, generate_multichange_sample


class Stand:
    def __init__(self, algos: List[Type[StreamingAlgo]], fuses: List[Fuse], node_distributions: List[Type[NodeDistribution]]):
        self.algos = algos
        self.fuses = fuses
        self.node_distributions = node_distributions

    def __run_test__(
            self,
            p: float,
            algo: Type[StreamingAlgo],
            fuse: Fuse,
            node_distribution_type: Type[NodeDistribution],
            n_nodes: int,
            sample: List[Tuple[int, float]],
            change_points: List[int]
    ) -> Dict[str, float]:
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
        tdrs = dict([(f'{algo.__name__} {node_distribution.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions])
        mdrs = dict([(f'{algo.__name__} {node_distribution.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions])
        fdrs = dict([(f'{algo.__name__} {node_distribution.__name__}', []) for algo in self.algos for node_distribution in self.node_distributions])

        sample, changepoints = generate_multichange_sample(100000)

        for n_nodes in rng:
            for node_distribution in self.node_distributions:
                for algo, fuse in zip(self.algos, self.fuses):
                    result = self.__run_test__(p, algo, fuse, node_distribution, n_nodes, sample, changepoints)
                    print(f'{algo.__name__} {node_distribution.__name__}: {result}')
                    tdrs[f'{algo.__name__} {node_distribution.__name__}'].append(result['TDR'])
                    mdrs[f'{algo.__name__} {node_distribution.__name__}'].append(result['MDR'])
                    fdrs[f'{algo.__name__} {node_distribution.__name__}'].append(result['FDR'])

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
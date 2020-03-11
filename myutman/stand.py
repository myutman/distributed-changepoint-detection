import json
import os
from datetime import datetime
from typing import Tuple, List, Dict, Type, Any, Optional

import numpy as np
from tqdm import tqdm

from myutman.fuse import Fuse
from myutman.node_distribution import NodeDistribution
from myutman.single_thread import StreamingAlgo
from myutman.stand_utils import calc_error


class Stand:
    def __init__(
        self,
        n_nodes,
        algo: Type[StreamingAlgo],
        account1_node_distribution: Type[NodeDistribution],
        account2_node_distribution: Type[NodeDistribution],
        fuse: Fuse,
        account1_algo_kwargs: Optional[Dict[str, Any]] = None,
        account2_algo_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.n_nodes = n_nodes
        self.algo = algo
        self.account1_node_distribution = account1_node_distribution(n_nodes)
        self.account2_node_distribution = account2_node_distribution(n_nodes)
        self.result_filename = f"outputs/algo={algo.__name__}_" \
                               f"account1_dist={account1_node_distribution.__name__}_" \
                               f"account2_dist={account2_node_distribution.__name__}_" \
                               f"nnodes={n_nodes}"
        self.fuse = fuse
        if account1_algo_kwargs is None:
            self.account1_algo_kwargs: Dict[str, Any] = {}
        else:
            self.account1_algo_kwargs: Dict[str, Any] = account1_algo_kwargs
        if account2_algo_kwargs is None:
            self.account2_algo_kwargs: Dict[str, Any] = {}
        else:
            self.account2_algo_kwargs: Dict[str, Any] = account2_algo_kwargs
        os.system('mkdir -p outputs')

    def test(
        self,
        p: float,
        sample: List[Tuple[Any, float]],
        change_points: List[int],
        change_ids: List[Tuple[int, int]],
        n_account1s,
        n_account2s
    ) -> Dict[str, float]:
        nodes1: np.ndarray = np.array([
            [
                self.algo(
                    p=p,
                    random_state=i + j * n_account1s,
                    **self.account1_algo_kwargs
                ) for i in range(n_account1s)
            ] for j in range(self.n_nodes)
        ])
        nodes2: np.ndarray = np.array([
            [
                self.algo(
                    p=p,
                    random_state=i + j * n_account2s,
                    **self.account2_algo_kwargs
                ) for i in range(n_account2s)
            ] for j in range(self.n_nodes)
        ])
        detected: List[int] = []
        detected_ids: List[Tuple[int, int]] = []
        for i, ((account1_id, account2_id), point) in tqdm(enumerate(sample)):
            detection = False
            detection_ids = [-1, -1]
            if n_account1s > 0:
                account1_node_id = self.account1_node_distribution.get_node_index(account1_id, account2_id)
                nodes1[account1_node_id][account1_id].process_element(point)
                if self.fuse(p, nodes1[:, account1_id]):
                    detection = True
                    detection_ids[0] = account1_id
                    for algo in nodes1[:, account1_id]:
                        algo.restart()
            if n_account2s > 0:
                account2_node_id = self.account2_node_distribution.get_node_index(account2_id, account1_id)
                nodes2[account2_node_id][account2_id].process_element(point)
                if self.fuse(p, nodes2[:, account2_id]):
                    detection = True
                    detection_ids[1] = account2_id
                    for algo in nodes2[:, account2_id]:
                        algo.restart()
            if detection:
                detected.append(i)
                detected_ids.append(tuple(detection_ids))

        error = calc_error(change_points, detected)#change_ids, detected, detected_ids)
        output_json = {
            "name": self.result_filename,
            "n_account1s": n_account1s,
            "n_account2s": n_account2s,
            "error": error,
            "sample": sample,
            "change_points": change_points,
            "change_ids": change_ids,
            "detected_change_points": detected,
            "detected_change_points_ids": detected_ids,
        }

        output_filename = f"{self.result_filename}_{datetime.now().timestamp()}.json"
        print(f"Result written to {output_filename}")
        with open(output_filename, 'w') as output_file:
            json.dump(output_json, output_file, ensure_ascii=False, indent=4)

        return error


"""
class Stand:
    def __init__(
        self,
        algos: List[Type[StreamingAlgo]],
        fuses: List[Fuse],
        account1_node_distributions: List[Type[NodeDistribution]],

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
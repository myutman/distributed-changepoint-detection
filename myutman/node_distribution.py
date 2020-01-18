import abc
import numpy as np

class NodeDistribution:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes

    @abc.abstractmethod
    def get_node_index(self, meta=None):
        pass


class RoundrobinNodeDistribution(NodeDistribution):
    def __init__(self, n_nodes: int):
        super(RoundrobinNodeDistribution, self).__init__(n_nodes)
        self.cur = 0

    def get_node_index(self, meta=None):
        ans = self.cur
        self.cur = (self.cur + 1) % self.n_nodes
        return ans


class DependentNodeDistribution(NodeDistribution):
    def __init__(self, n_nodes: int):
        super(DependentNodeDistribution, self).__init__(n_nodes)

    def get_node_index(self, meta=None):
        return hash(meta) % self.n_nodes


class VariableNodeDistribution(NodeDistribution):
    def _init_(self, n_nodes: int, alpha: float, state: int = 0):
        super(VariableNodeDistribution, self).__init__(n_nodes)
        self.alpha = alpha
        self.random_state = np.random.RandomState(state)

    def get_node_index(self, meta=None):
        if self.random_state.binomial(1, self.alpha) == 1:
            return hash(meta) % self.n_nodes
        else:
            return self.random_state.choice(self.n_nodes)

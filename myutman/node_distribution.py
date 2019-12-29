import abc


class NodeDistribution:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

    @abc.abstractmethod
    def get_node_index(self, meta=None):
        pass


class RoundrobinNodeDistribution(NodeDistribution):
    def __init__(self, n_nodes):
        super(RoundrobinNodeDistribution, self).__init__(n_nodes)
        self.cur = 0

    def get_node_index(self, meta=None):
        ans = self.cur
        self.cur = (self.cur + 1) % self.n_nodes
        return ans


class DependentNodeDistribution(NodeDistribution):
    def __init__(self, n_nodes):
        super(DependentNodeDistribution, self).__init__(n_nodes)

    def get_node_index(self, meta=None):
        return hash(meta) % self.n_nodes

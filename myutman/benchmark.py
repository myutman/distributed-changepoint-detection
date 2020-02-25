from myutman.fuse import FuseForWindowAlgo
from myutman.node_distribution import RoundrobinNodeDistribution, DependentNodeDistribution
from myutman.stand import Stand
from myutman.stand_utils import generate_problem_sample
from myutman.window_algo import WindowStreamingAlgo

if __name__ == '__main__':
    #stand = Stand([WindowStreamingAlgo], [FuseForWindowAlgo()], [RoundrobinNodeDistribution, DependentNodeDistribution])
    #stand.compare_distibuted_algorithms_plots()
    generate_problem_sample()

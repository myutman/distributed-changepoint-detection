from myutman.stand import run_test, generate_multichange_sample, compare_distibuted_algorithms_plots
from myutman.window_algo import WindowStreamingAlgo, WindowRoundrobinStreamingAlgo, WindowDependentStreamingAlgo

if __name__ == '__main__':

    sample, change_points = generate_multichange_sample(100000)

    #algo = WindowStreamingAlgo(0.01, 30, [(71, 93), (80, 90), (65, 79)])
    #run_test(algo, sample, change_points)

    #algo1 = WindowRoundrobinStreamingAlgo(0.01, 5, 30, [(71, 93), (80, 90), (65, 79)])
    #run_test(algo1, sample, change_points)

    #algo2 = WindowDependentStreamingAlgo(0.01, 2)
    #result = run_test(algo2, sample, change_points)
    #print(result)

    compare_distibuted_algorithms_plots([WindowDependentStreamingAlgo, WindowRoundrobinStreamingAlgo])

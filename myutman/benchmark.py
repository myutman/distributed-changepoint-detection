from myutman.stand import run_test, run_test_dependent
from myutman.window_algo import WindowStreamingAlgo, WindowRoundrobinStreamingAlgo

if __name__ == '__main__':
    #algo = WindowStreamingAlgo(0.01, 30, [(71, 93), (80, 90), (65, 79)])
    #run_test(algo)

    algo1 = WindowRoundrobinStreamingAlgo(0.01, 5, 30, [(71, 93), (80, 90), (65, 79)])
    run_test_dependent(algo1)

    algo2 = WindowRoundrobinStreamingAlgo(0.01, 5, 30, [(71, 93), (80, 90), (65, 79)])
    run_test_dependent(algo2)
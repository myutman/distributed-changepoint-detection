from myutman.single_thread import StreamingAlgo

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def generate_multichange_sample(size, n_modes=13, probs=None, tau=1000, tau_noise=1, delta=10, delta_noise=0.1):
    sample = []
    change_points = []
    mu = np.arange(n_modes, dtype=np.float64)
    limit = int(np.random.normal(tau, tau_noise))
    if probs is None:
        probs = 1 / np.arange(1, n_modes + 1)
        probs /= probs.sum()
    for i in range(size):
        if i == limit:
            limit += int(np.random.normal(tau, tau_noise))
            mu += np.random.normal(delta, delta_noise, size=n_modes)
            change_points.append(i)
        mode = np.random.choice(n_modes, p=probs)
        sample.append((mode, np.random.normal(mu[mode], 1)))
    return sample, change_points


def calc_error(change_points, detected):
    correct_detection = 0
    false_detection = 0
    missed_detection = 0

    j = 0
    for i, d in enumerate(change_points):
        # skip all false detections before change_points[i]
        while j < len(detected) and detected[j] < d:
            false_detection += 1
            j += 1

        # if change_points[i] and change_points[i+1] occur without detection then a missed detection
        if j == len(detected) or (i + 1 < len(change_points) and change_points[i + 1] <= detected[j]):
            missed_detection += 1
        else:
            correct_detection += 1
            j += 1

    return {'TDR': correct_detection / len(change_points), 'MDR': 1 - correct_detection / len(change_points), 'FDR': 1 - correct_detection / len(detected)}


def run_test(algo: StreamingAlgo, sample, change_points):
    detected = []
    for i, (point, meta) in tqdm(enumerate(sample)):
        algo.process_element(point, meta)
        if algo.test():
            detected.append(i)
            algo.restart()

    return calc_error(change_points, detected)


def compare_distibuted_algorithms_plots(algos):
    tdrs = dict([(algo.__name__, []) for algo in algos])
    mdrs = dict([(algo.__name__, []) for algo in algos])
    fdrs = dict([(algo.__name__, []) for algo in algos])

    sample, changepoints = generate_multichange_sample(100000)
    for n_nodes in range(1, 14):
        for algo in algos:
            result = run_test(algo(0.01, n_nodes), sample, changepoints)
            print(result)
            tdrs[algo.__name__].append(result['TDR'])
            mdrs[algo.__name__].append(result['MDR'])
            fdrs[algo.__name__].append(result['FDR'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(nrows=3, ncols=1, sharex=True, sharey=True)
    for lst in list(tdrs.values()):
        ax[0].plot(range(1, 14), lst)
    ax[0].legend(list(tdrs.keys()))
    ax[0].set_title('TDR')

    for lst in list(mdrs.values()):
        ax[1].plot(range(1, 14), lst)
    ax[1].legend(list(mdrs.keys()))
    ax[1].set_title('MDR')

    for lst in list(fdrs.values()):
        ax[2].plot(range(1, 14), lst)
    ax[2].legend(list(fdrs.keys()))
    ax[2].set_title('FDR')

    plt.show()
from myutman.single_thread import StreamingAlgo

from tqdm import tqdm
import numpy as np


def generate_multichange_sample(size, n_modes=5, probs=None, tau=1000, tau_noise=1, delta=10, delta_noise=0.1):
    sample = []
    change_points = []
    mu = np.arange(5, dtype=np.float64)
    limit = int(np.random.normal(tau, tau_noise))
    if probs is None:
        probs = np.ones(n_modes) / n_modes
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


def run_test(algo: StreamingAlgo):
    sample, change_points = generate_multichange_sample(100000)
    detected = []
    for i, (point, meta) in tqdm(enumerate(sample)):
        algo.process_element(point, meta)
        if algo.test():
            detected.append(i)
            algo.restart()

    print(calc_error(change_points, detected))

def run_test_dependent(algo: StreamingAlgo):
    sample, change_points = generate_multichange_sample(100000, probs=[0.1, 0.2, 0.3, 0.2, 0.2])
    detected = []
    for i, (point, meta) in tqdm(enumerate(sample)):
        algo.process_element(point, meta)
        if algo.test():
            detected.append(i)
            algo.restart()

    print(calc_error(change_points, detected))
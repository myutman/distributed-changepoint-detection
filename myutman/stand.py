from myutman.single_thread import StreamingAlgo

from tqdm import tqdm
import numpy as np

def generate_multichange_sample(size, tau=1000, tau_noise=1, delta=10, delta_noise=0.1):
    sample = []
    change_points = []
    mu = 0
    limit: int = int(np.random.normal(tau, tau_noise))
    for i in range(size):
        if i == limit:
            limit += int(np.random.normal(tau, tau_noise))
            #print(limit)
            mu += np.random.normal(delta, delta_noise)
            change_points.append(i)
        sample.append(np.random.normal(mu, 1))
    return sample, change_points


def calc_error(change_points, detected, tol=100):
    j = 0
    t = 0
    for d in detected:
        #print(d)
        while j < len(change_points) and change_points[j] < d - tol:
            #print(change_points[j])
            j += 1
        if j < len(change_points) and change_points[j] < d:
            #print(change_points[j])
            t += 1
            j += 1
    return {'TDR': t / len(change_points), 'MDR': 1 - t / len(change_points), 'FDR': (len(detected) - t) / len(detected)}


def run_test(algo: StreamingAlgo):
    sample, change_points = generate_multichange_sample(100000)
    detected = []
    for i, point in tqdm(enumerate(sample)):
        algo.process_element(point)
        if algo.test():
            detected.append(i)
            algo.restart()
        #if i % 5000 == 0:
        #    print(i)

    print(calc_error(change_points, detected))
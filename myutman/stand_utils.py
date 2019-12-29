from typing import Tuple, List, Dict, Type
import numpy as np

def generate_multichange_sample(
        size: int,
        n_modes: int = 13,
        probs: List[float] = None,
        tau: float = 1000,
        tau_noise: float = 1,
        delta: float = 10,
        delta_noise: float = 0.1
) -> Tuple[List[Tuple[int, float]], List[int]]:
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


def calc_error(change_points: List[int], detected: List[int]) -> Dict[str, float]:
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

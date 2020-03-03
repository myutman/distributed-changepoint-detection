from typing import List, Dict
import scipy.stats as scp
import matplotlib.pyplot as plt

def calc_error(change_points: List[int], detected: List[int]) -> Dict[str, float]:
    correct_detection = 0
    latency = 0

    j = 0
    for i, d in enumerate(change_points):
        # skip all false detections before change_points[i]
        while j < len(detected) and detected[j] < d:
            j += 1

        # if change_points[i] and change_points[i+1] occur without detection then a missed detection
        if j < len(detected) and (i + 1 >= len(change_points) or change_points[i + 1] > detected[j]):
            latency += detected[j] - change_points[i]
            correct_detection += 1
            j += 1

    return {
        'TDR': correct_detection / len(change_points),
        'MDR': 1 - correct_detection / len(change_points),
        'FDR': 1 - correct_detection / len(detected),
        'latency': latency / correct_detection
    }


def compare_mdrs(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1, legend2):
    mdrs1 = [res['MDR'] for res in result1]
    mdrs2 = [res['MDR'] for res in result2]
    wx = scp.wilcoxon(mdrs1, mdrs2, alternative='less')
    print(f"Results of Wilcoxon T-test of {legend1} MDR to be less then {legend2} MDR is {wx}")


def compare_fdrs(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1, legend2):
    fdrs1 = [res['FDR'] for res in result1]
    fdrs2 = [res['FDR'] for res in result2]
    wx = scp.wilcoxon(fdrs1, fdrs2, alternative='less')
    print(f"Results of Wilcoxon T-test of {legend1} FDR to be less then {legend2} MDR is {wx}")


def compare_latencies(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1, legend2):
    latencies1 = [res['latency'] for res in result1]
    latencies2 = [res['latency'] for res in result2]
    plt.boxplot([latencies1, latencies2], labels=[legend1, legend2])
    plt.ylabel('Mean latency, iterations')
    plt.show()

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp


def calc_error(change_points: List[int], detected: List[int]) -> Dict[str, float]:
    changes = 0
    detections = 0
    missed_changes = 0
    true_detections = 0
    false_detections = 0
    late_detections = 0

    currently_detecting_change = 0
    latency = 0

    j = 0
    previous_was_missed = False
    for i, d in enumerate(change_points):
        changes += 1
        # skip all false detections before change_points[i]
        while j < len(detected) and detected[j] < d:
            detections += 1
            false_detections += 1
            j += 1

        # if change_points[i] and change_points[i+1] occur without detection then a missed detection
        if j == len(detected) or (i + 1 < len(change_points) and change_points[i + 1] <= detected[j]):
            missed_changes += 1
            if not previous_was_missed:
                currently_detecting_change = d
                previous_was_missed = True
        else:
            detections += 1
            if previous_was_missed:
                latency += detected[j] - currently_detecting_change
                late_detections += 1
            else:
                latency += detected[j] - d
                true_detections += 1
            previous_was_missed = False
            j += 1
    while j < len(detected):
        detections += 1
        false_detections += 1
        j += 1

    return {
        'changes': changes,
        'detections': detections,
        'true_detections': true_detections,
        'false_detections': false_detections,
        'late_detections': late_detections,
        'total_errored_detections': false_detections + late_detections,
        'missed_changes': missed_changes,
        'TDR': (true_detections / changes) if changes > 0 else None,
        'MDR': (1 - true_detections / changes) if changes > 0 else None,
        'FDR': (false_detections / detections) if detections > 0 else None,
        'latency': (latency / (true_detections + late_detections)) if (true_detections + late_detections) > 0 else None
    }


"""def calc_error(change_points: List[int], change_ids: List[Tuple[int, int]], detected: List[int], detected_ids: List[Tuple[int, int]]) -> Dict[str, float]:
    account1_detection = 0
    account2_detection = 0

    account1_change = 0
    account2_change = 0

    correct_account1_detection = 0
    correct_account2_detection = 0

    account1_cross_detection = 0
    account2_cross_detection = 0

    account1_latency = 0
    account2_latency = 0

    j = 0
    for i, (d, (account1_id, account2_id)) in enumerate(zip(change_points, change_ids)):
        if account1_id != -1:
            account1_change += 1
        if account2_id != -1:
            account2_change += 1

        # skip all false detections before change_points[i]
        while j < len(detected) and detected[j] < d:
            detected_account1_id, detected_account2_id = detected_ids[j]
            if detected_account1_id != -1:
                account1_detection += 1
            if detected_account2_id != -1:
                account2_detection += 1
            j += 1

        already_detected = False

        # if change_points[i] and change_points[i+1] occur without detection then a missed detection
        while j < len(detected) and (i + 1 >= len(change_points) or change_points[i + 1] >= detected[j]):
            detected_account1_id, detected_account2_id = detected_ids[j]
            if detected_account1_id != -1:
                account1_detection += 1
            if detected_account2_id != -1:
                account2_detection += 1
            if detected_account1_id != -1:
                if detected_account1_id == account1_id and not already_detected:
                    already_detected = True
                    account1_latency += detected[j] - d
                    correct_account1_detection += 1
                elif account1_id == -1:
                    account1_cross_detection += 1
            if detected_account2_id != -1:
                if detected_account2_id == account2_id and not already_detected:
                    already_detected = True
                    account2_latency += detected[j] - d
                    correct_account2_detection += 1
                elif account2_id == -1:
                    account2_cross_detection += 1
            j += 1

    return {
        'account1_TDR': correct_account1_detection / account1_change if account1_change > 0 else None,
        'account1_MDR': 1 - correct_account1_detection / account1_change if account1_change > 0 else None,
        'account1_FDR': 1 - correct_account1_detection / (account1_detection - account1_cross_detection) if account1_detection - account1_cross_detection > 0 else None,
        'account1_CDR': account1_cross_detection / account2_change if account2_change > 0 else None,
        'account2_TDR': correct_account2_detection / account2_change if account2_change > 0 else None,
        'account2_MDR': 1 - correct_account2_detection / account2_change if account2_change > 0 else None,
        'account2_FDR': 1 - correct_account2_detection / (account2_detection - account2_cross_detection) if account2_detection - account2_cross_detection > 0 else None,
        'account2_CDR': account2_cross_detection / account1_change if account1_change > 0 else None,
        'account1_False': account1_detection - correct_account1_detection - account1_cross_detection,
        'account2_False': account2_detection - correct_account2_detection - account2_cross_detection,
        'account1_latency': account1_latency / correct_account1_detection if correct_account1_detection > 0 else None,
        'account2_latency': account2_latency / correct_account2_detection if correct_account2_detection > 0 else None
    }"""


def compare_mdrs(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1: str, legend2: str):
    mdrs1 = [res['MDR'] for res in result1]
    mdrs2 = [res['MDR'] for res in result2]
    wx = scp.wilcoxon(mdrs1, mdrs2, alternative='less')
    print(f"Results of Wilcoxon T-test of {legend1} MDR to be less then {legend2} MDR is {wx}")


def compare_fdrs(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1: str, legend2: str):
    fdrs1 = [res['FDR'] for res in result1]
    fdrs2 = [res['FDR'] for res in result2]
    wx = scp.wilcoxon(fdrs1, fdrs2, alternative='less')
    print(f"Results of Wilcoxon T-test of {legend1} FDR to be less then {legend2} FDR is {wx}")


def compare_latencies(result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1: str, legend2: str):
    latencies1 = [res['latency'] for res in result1]
    latencies2 = [res['latency'] for res in result2]
    plt.boxplot([latencies1, latencies2], labels=[legend1, legend2])
    plt.ylabel('Mean latency, iterations')
    plt.show()


def compare_vals(val_name: str, result1: List[Dict[str, float]], result2: List[Dict[str, float]], legend1, legend2):
    vals1 = [res[val_name] for res in result1]
    vals2 = [res[val_name] for res in result2]
    wx = scp.wilcoxon(vals1, vals2, alternative='two-sided')
    print(f"Results of Wilcoxon T-test of {legend1} {val_name} to be less then {legend2} {val_name} is {wx}")


def show_boxplots(val_name: str, results: List[List[Dict[str, float]]], legends: List[str]):
    valss = [[res[val_name] for res in result if res[val_name] is not None] for result in results]
    plt.boxplot(valss, labels=legends)
    plt.ylabel(val_name)
    plt.show()


def show_plot_by_p_levels(val_name: str, resultss: List[List[List[Dict[str, float]]]], legends: List[str]):
    valss = [[np.median([res[val_name] for res in result if res[val_name] is not None]) for result in results] for results in resultss]
    for vals in valss:
        #plt.boxplot(vals, labels=['p=0.01', 'p=0.05', 'p=0.1'])
        plt.plot(['p=0.01', 'p=0.05', 'p=0.1'], vals)
    plt.legend(legends)
    plt.ylabel(val_name)
    plt.show()


def show_boxplots_by_p_levels(val_name: str, resultss: List[List[List[Dict[str, float]]]], legends: List[str]):
    valss = np.array([[[res[val_name] for res in result if res[val_name] is not None] for result in results] for
        results in resultss]).transpose((1, 0, 2))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    print(valss.shape)
    for i, (vals, p_level) in enumerate(zip(valss, [0.01, 0.05, 0.1])):
        print(vals.shape)
        axs[i].boxplot([*vals], labels=legends)
        axs[i].axhline(y=p_level)
    plt.ylabel(val_name)
    plt.show()

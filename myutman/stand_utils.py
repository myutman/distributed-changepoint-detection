from typing import List, Dict, Tuple
import scipy.stats as scp
import matplotlib.pyplot as plt


def calc_error(change_points: List[int], change_ids: List[Tuple[int, int]], detected: List[int], detected_ids: List[Tuple[int, int]]) -> Dict[str, float]:
    client_detection = 0
    terminal_detection = 0

    client_change = 0
    terminal_change = 0

    correct_client_detection = 0
    correct_terminal_detection = 0

    client_cross_detection = 0
    terminal_cross_detection = 0

    client_latency = 0
    terminal_latency = 0

    j = 0
    for i, (d, (client_id, terminal_id)) in enumerate(zip(change_points, change_ids)):
        if client_id != -1:
            client_change += 1
        if terminal_id != -1:
            terminal_change += 1

        # skip all false detections before change_points[i]
        while j < len(detected) and detected[j] < d:
            detected_client_id, detected_terminal_id = detected_ids[j]
            if detected_client_id != -1:
                client_detection += 1
            if detected_terminal_id != -1:
                terminal_detection += 1
            j += 1

        already_detected = False

        # if change_points[i] and change_points[i+1] occur without detection then a missed detection
        while j < len(detected) and (i + 1 >= len(change_points) or change_points[i + 1] >= detected[j]):
            detected_client_id, detected_terminal_id = detected_ids[j]
            if detected_client_id != -1:
                client_detection += 1
            if detected_terminal_id != -1:
                terminal_detection += 1
            if detected_client_id != -1:
                if detected_client_id == client_id and not already_detected:
                    already_detected = True
                    client_latency += detected[j] - d
                    correct_client_detection += 1
                elif client_id == -1:
                    client_cross_detection += 1
            if detected_terminal_id != -1:
                if detected_terminal_id == terminal_id and not already_detected:
                    already_detected = True
                    terminal_latency += detected[j] - d
                    correct_terminal_detection += 1
                elif terminal_id == -1:
                    terminal_cross_detection += 1
            j += 1

    return {
        'client_TDR': correct_client_detection / client_change if client_change > 0 else None,
        'client_MDR': 1 - correct_client_detection / client_change if client_change > 0 else None,
        'client_FDR': 1 - correct_client_detection / (client_detection - client_cross_detection) if client_detection - client_cross_detection > 0 else None,
        'client_CDR': client_cross_detection / terminal_change if terminal_change > 0 else None,
        'terminal_TDR': correct_terminal_detection / terminal_change if terminal_change > 0 else None,
        'terminal_MDR': 1 - correct_terminal_detection / terminal_change if terminal_change > 0 else None,
        'terminal_FDR': 1 - correct_terminal_detection / (terminal_detection - terminal_cross_detection) if terminal_detection - terminal_cross_detection > 0 else None,
        'terminal_CDR': terminal_cross_detection / client_change if client_change > 0 else None,
        'client_False': client_detection - correct_client_detection - client_cross_detection,
        'terminal_False': terminal_detection - correct_terminal_detection - terminal_cross_detection,
        'client_latency': client_latency / correct_client_detection if correct_client_detection > 0 else None,
        'terminal_latency': terminal_latency / correct_terminal_detection if correct_terminal_detection > 0 else None
    }


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

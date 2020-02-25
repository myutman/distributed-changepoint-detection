from typing import List, Dict

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

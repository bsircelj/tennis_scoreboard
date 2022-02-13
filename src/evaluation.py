import numpy as np
import enchant
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def evaluate(truth_array, predicted_array):
    """
    Evaluates the model performance. The methods receives two arrays that correspond to true and predicted detection
    results. Each entry must be in the same structure as is the structure for specific frames in the file
    top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json
    Example of the entry:
    {"bbox": [193.06, 905.52, 549.29, 995.99],
                            "serving_player": "name_2",
                            "name_1": "Thiem",
                            "name_2": "Khachanov",
                            "score_1": "4-1-0", "score_2": "6-2-40"}

    Metrics calculated and displayed:
        - Intersection over union for the bounding boxes (IOU). The resulting plot shows how many of the images have
            higher IOU score than the chosen percentage. The percentages are ranging from 0.5 to 0.95 in steps of 0.05.
        - Correct matches for names and scores
        - Levenshtein distance for incorrect names and scores


    :param truth_array: array of detection results
    :param predicted_array: array of ground truths
    """
    iou_percentage = np.arange(0.5, 0.95, 0.05)
    iou_count = np.zeros(np.shape(iou_percentage))

    correct_names = 0
    correct_scores = 0
    names_levenshtein_sum = 0
    score_levenshtein_sum = 0
    false_negative = 0
    false_positive = 0
    length = 0

    for truth, predicted in zip(truth_array, predicted_array):
        # Scoreboard without detection FN
        if truth is not None and predicted is None:
            false_negative += 1

        # Detection without a scoreboard FP
        elif truth is None and predicted is not None:
            false_positive += 1

        # Both present TP
        elif truth is not None and predicted is not None:
            length += 1
            iou = intersection_over_union(truth['bbox'], predicted['bbox'])
            for i, percentage in enumerate(iou_percentage):
                if iou >= percentage:
                    iou_count[i] += 1

            for n1, n2 in [(truth['name_1'], predicted['name_1']), (truth['name_2'], predicted['name_2'])]:
                if n1 == n2:
                    correct_names += 1
                else:
                    names_levenshtein_sum += enchant.utils.levenshtein(n1, n2)

            for n1, n2 in [(truth['score_1'], predicted['score_1']), (truth['score_2'], predicted['score_2'])]:
                if n1 == n2:
                    correct_scores += 1
                else:
                    score_levenshtein_sum += enchant.utils.levenshtein(n1, n2)

    # average over all entries
    names_levenshtein_sum /= (length * 2 - correct_names)
    score_levenshtein_sum /= (length * 2 - correct_scores)
    correct_names /= (length * 2)
    correct_scores /= (length * 2)
    false_negative /= len(truth_array)
    false_positive /= len(truth_array)
    title = f'Cor names: {correct_names:.2f} Cor score: {correct_scores:.2f} Lev names: {names_levenshtein_sum:.2f}' + \
            f' Lev score: {score_levenshtein_sum:.2f} FN: {false_negative:.2f} FP: {false_positive:.2f}'

    plt.figure(figsize=(12, 8))
    plt.bar(iou_percentage, iou_count, width=0.04)
    plt.xticks(iou_percentage, [f'{x:3.2f}' for x in iou_percentage])
    plt.xlabel("min IOU percentage area")
    plt.ylabel("detection count")
    plt.title(title)
    plt.show()


def intersection_over_union(box1, box2):
    """
    The box is defined by four points. First is in the upper left corner with points x1 and y1. The second point is in
    the lower right corner with coordinates x2, y2. The structure of the input array must be the following:
    [x1, y1, x2, y2]

    :param box1: coordinates of the first bounding box
    :type box1: list<float>
    :param box2: coordinates of the second bounding box
    :type box2: list<float>
    :return: intersection over union for both boxes
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection / float(area1 + area2 - intersection)

    # return the intersection over union value
    return iou

import cv2
import json
import numpy as np
from video_processing import sample_stream
from yolo_utilities import YoloModel
from evaluation import evaluate


def run():
    with open("./data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json", "r") as file:
        annotations = json.load(file)
    annotated_frames = np.array(list(annotations.keys())).astype(int)
    image_stream = sample_stream("./data/matches/47/val")
    model = YoloModel("./data/models/best.pt", 0.4)
    predicted = np.array([])
    truth = np.array([])
    for frame_no, filename, image in image_stream:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predicted = np.concatenate((predicted, [model(image)]))
        if int(frame_no) in annotated_frames:
            truth = np.concatenate((truth, [annotations[frame_no]]))
        else:
            truth = np.concatenate((truth, [None]))

    evaluate(truth, predicted)


if __name__ == '__main__':
    run()

#!/bin/bash

YOLO_ROOT=/home/beno/PycharmProjects/yolov5
source "$YOLO_ROOT/venv/bin/activate"

python "$YOLO_ROOT/train.py" --img 720 --batch 4 --epochs 150 --data dataset700.yaml --weights yolov5s.pt --cache
#!/bin/bash

YOLO_ROOT=/home/beno/PycharmProjects/yolov5
source "$YOLO_ROOT/venv/bin/activate"

python "$YOLO_ROOT/val.py" --img 480 --data dataset700.yaml --weights ../data/models/best.pt
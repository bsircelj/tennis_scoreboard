from video_processing import load_images
import os
import numpy as np
from PIL import Image
import cv2
from text_extraction import read_text
import yolov5


class YoloModel:
    """
    A class that serves as a wrapper for the YOLOv5[1] object detection model and tesseract[2] OCR. It supports loading
    already trained YOLO model, making predictions with it, calling the tesseract to extract the text data and packaging
    the results in to the specified form


    [1] YOLOv5 https://github.com/ultralytics/yolov5
    [2] pytesseract https://github.com/madmaze/pytesseract
    """

    def __init__(self, model_location, confidence_threshold=0):
        """
        :param model_location: location of the YOLOv5 model
        :param confidence_threshold: The threshold for counting the detections as valid
        """
        self.model = yolov5.load(model_location)
        self.confidence_threshold = confidence_threshold

    def predict(self, image):
        """
        The image must be an array in RGB color scheme with the dimensions (height, width, 3)
        :param image: image saves as an array
        :return: dictionary with results or None which represents no detection
        """
        results = self.model(image)
        predictions = results.pred[0].cpu().numpy()
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        if categories.size == 0:
            return None

        # if the score is not above the threshold the detection is discarded
        if scores[0] < self.confidence_threshold:
            return None

        boxes = np.squeeze(predictions[:, :4])
        if len(np.shape(boxes)) == 2:
            boxes = boxes[0]

        names, scores = read_text(image, boxes)
        returned_dict = {"bbox": list(boxes),
                         "serving_player": "name_1" if int(categories[0]) == 0 else "name_2",
                         "name_1": names[0],
                         "name_2": names[1],
                         "score_1": scores[0],
                         "score_2": scores[1]}

        return returned_dict

    def __call__(self, image):
        return self.predict(image)


def bbox_to_yolo(entry, image_width=1920, image_height=1080):
    """
    Extracts the serving player and bounding box and transforms it to the YOLOv5 format.

    The box defined by upper left (x1,y1) and lower right (x2,y2) is transformed to the YOLOv5 format which is
    represented of coordinates of the center of the box (x_center,y_center) along with the boxes height and width
    scaled to 0..1 for x and y dimension.

    :param entry: dictionary with the annotation
    :param image_width: width of the input image
    :param image_height: height of the input image
    :return: tuple of (class, x_center, y_center, box width, box height)
    """
    x1, y1, x2, y2 = entry["bbox"]
    class_no = 0 if entry["serving_player"] == "name_1" else 1
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height

    return class_no, x_center, y_center, width, height


def yolo_to_bbox(x_center, y_center, width, height, image_width=1920, image_height=1080):
    """
    Does the transformation of YOLOv5 format to corner bounding box format opposite as in the :func: `~bbox_to_yolo`


    :param x_center: x coordinate of bbox center
    :param y_center: y coordinate of bbox center
    :param width: bbox width
    :param height: bbox height
    :param image_width: width of the target image
    :param image_height: height of the target image
    :return:
    """
    x1 = (x_center - width / 2) * image_width
    x2 = (x_center + width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    y2 = (y_center + height / 2) * image_height
    return x1, x2, y1, y2


def create_yolo_dataset(annotations, train_folder, val_folder, test_folder, output_root):
    """
    Creates dataset in the format that is specified by YOLOv5

    :param annotations: image annotations
    :param train_folder: location of images for training
    :param val_folder: location of images for validation
    :param test_folder: location of images for testing
    :param output_root: location of resulting processed dataset
    """
    annotated_frames = np.array(list(annotations.keys())).astype(int)

    for load_folder, save_folder in [(train_folder, "train"),
                                     (val_folder, "val"),
                                     (test_folder, "test")]:
        if not os.path.exists(f'{output_root}/images/{save_folder}'):
            os.makedirs(f'{output_root}/images/{save_folder}')
            os.makedirs(f'{output_root}/labels/{save_folder}')

        frame_numbers, filenames, video_array = load_images(load_folder)
        for i, _ in enumerate(video_array):
            video_array[i] = cv2.cvtColor(video_array[i], cv2.COLOR_BGR2RGB)

        img_no = 0
        for i, f in enumerate(frame_numbers):
            print(f"Processing image {img_no}")
            filename = f'{img_no:04}'
            if int(f) in annotated_frames:
                with open(f'{output_root}/labels/{save_folder}/{filename}.txt', 'w') as file:
                    file.write('{0} {1:.8f} {2:.8f} {3:.8f} {4:.8f}\n'.format(*bbox_to_yolo(annotations[str(f)])))
            else:
                print(f'Skipping label for {filename}')
            image = Image.fromarray(video_array[i])
            image.save(f'{output_root}/images/{save_folder}/{filename}.jpg')
            img_no += 1
    dataset_name = output_root.split("/")[-1]

    with open(f'{output_root}/{dataset_name}.yaml', 'w') as file:
        file.write(f"path: ../datasets/{dataset_name}\n")
        file.write("train: images/train\n")
        file.write("val: images/val\n")
        file.write("test: images/test\n")
        file.write("nc: 2\n")
        file.write("names: [\'name_1\', \'name_2\']\n")


def check_yolo_dataset(image_folder, label_folder, output_folder):
    """
    Draws bounding boxes on all the images. Drawn bounding box is blue for "name_1" and green for "name_2". The images
    and labels must be in the YOLOv5 format. This method is used for double checking the dataset before training.

    :param image_folder: location of images
    :param label_folder: location of labels
    :param output_folder: location of resulting images
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            image = cv2.imread(f'{image_folder}/{file}', cv2.IMREAD_COLOR)
            new_image = image
            if os.path.exists(f'{label_folder}/{file.split(".")[0]}.txt'):
                with open(f'{label_folder}/{file.split(".")[0]}.txt', 'r') as f:
                    line = f.read()
                    class_no, x_center, y_center, width, height = [float(x) for x in line.split()]
                    bbox = yolo_to_bbox(x_center, y_center, width, height)
                    x1, x2, y1, y2 = np.array(bbox).astype(np.uint32)
                    new_image = cv2.rectangle(image, (x1, y1), (x2, y2),
                                              (255, 0, 0) if class_no == 0 else (0, 255, 0),
                                              3)
            else:
                print(f'Missing {label_folder}/{file.split(".")[0]}.txt')
            cv2.imwrite(f'{output_folder}/{file}', new_image)

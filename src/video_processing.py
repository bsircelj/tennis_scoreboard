import numpy as np
import os
import pytesseract
import cv2
import re


def video_stream(location):
    """
    Returns an generator that processes an video file and returns is frame by frame along with frame_no

    :param location: location of a video file
    :return: frame number along with the frame stored in an array
    :return type: (int, array)
    """
    capture = cv2.VideoCapture(location)

    if not capture.isOpened():
        print("Error opening the file")

    success = True
    frame_no = 0
    while capture.isOpened() and success:
        success, frame = capture.read()
        if success:
            frame_no += 1
            yield frame_no, frame

    capture.release()
    print("Video processed")


def sample_stream(folder):
    """
    Returns an generator that processes the images in a folder. The images must be named {frame_no}-*.png

    :param folder: location of the image folder
    :return: returns the frame number, file name and the image stored in an array
    :return type: (int, str, array<int>)
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            if ".png" in file:
                frame_no = file.split("-")[0]
                image = cv2.imread(f'{folder}/{file}', cv2.IMREAD_COLOR)
                yield frame_no, file, image
        break


def load_images(folder, height=1080, width=1920):
    """
    Loads all the images that are located in a specified folder and returns them in an array. The array is the shape of
    n * height * width * 3 where the n is the number of images in the folder. The n-th entry in the frame_number list
    and the filenames list correspond to the n-th entry in the video_array

    :param folder: location of the folder
    :type folder: string
    :param height: image height
    :param width: image width
    :return: a tuple of frame numbers, filenames and images
    :return type: (list<int>, list<string>, array<int>)
    """
    for root, dirs, files in os.walk(folder):
        count = len(files)
        video_array = np.empty((count, height, width, 3), np.dtype('uint8'))
        frame_count = 0
        filenames = []
        frame_numbers = []
        for file in files:
            if ".png" in file:
                frame_no = file.split("-")[0]
                video_array[frame_count] = cv2.imread(f'{folder}/{file}', cv2.IMREAD_COLOR)
                frame_count += 1
                filenames.append(file)
                frame_numbers.append(frame_no)

        return frame_numbers, filenames, video_array


def get_each_match_sample(location, save_path, frequency):
    """
    Loads a video frame by frame and for every n-th frame, where n is defined by the frequency argument, extracts the
    match number which is in the upper right corner using tesseract OCR and saves the image with the filename
    {frame number}-{match number}.png to the folder specified in the save_path

    :param location: path to the video from where the images are loaded
    :param save_path: location of the folder where the generated images are saved
    :param frequency: how often the frame is extracted and saved
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    get_frame = video_stream(location)
    part_bbox = [1650, 40, 1900, 225]
    current_match = []

    for frame_no, frame in get_frame:
        if frame_no % frequency != 0:
            continue
        match_no = frame[part_bbox[1]:part_bbox[3], part_bbox[0]:part_bbox[2]]
        match_no = cv2.cvtColor(match_no, cv2.COLOR_BGR2GRAY)
        match_no = match_no < 160
        match_no = match_no.astype(np.uint8) * 255
        results = pytesseract.image_to_string(match_no, config='--psm 13')
        results = re.sub(r"[^0-9]+", "", results)
        if results != '':
            try:
                filename = f'{save_path}/{frame_no}-{int(results)}.png'
                cv2.imwrite(filename, frame)
                print(f'Saving {filename}')
                current_match.append(int(results))
            except ValueError:
                pass

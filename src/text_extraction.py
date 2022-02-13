import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import matplotlib

matplotlib.use('TkAgg')


def read_text(raw_image, bbox):
    """
    Reads text using tesseract[1] OCR and returns names and scores for both players

    [1] pytesseract https://github.com/madmaze/pytesseract

    :param raw_image: raw input image
    :param bbox: location of text bounding box
    :return: two arrays [name_1, name_2] and [score_1, score2]
    :return type: list<string>, list<string>
    """
    image1, image2 = prepare_image(raw_image, bbox)
    name = ["", ""]
    score = ["", ""]
    for i, image in enumerate([image1, image2]):
        image = Image.fromarray(image)
        text = pytesseract.image_to_string(image, config='--psm 13')

        # assumes names don't contain numbers so the from first number marks where the score begins
        score_index = re.search(r"\d", text)
        if score_index:
            score_index = score_index.start()
        else:
            score_index = -1
        name[i] = text[:score_index]
        score_array = text[score_index:]

        # assumes all the names are anglicized and don't contain any non-English letters. Allowed characters are also
        # space " " and dot "."
        name[i] = re.sub(r"[^a-zA-Z. ]*", "", name[i]).lstrip().rstrip()

        # score can contain only numbers and words [Ad | All | Love]
        score_array = re.findall(r"\d+|Ad|All|Love", score_array)
        for s in score_array:
            score[i] = f"{score[i]}-{s}"
        score[i] = score[i][1:]

    return name, score


def prepare_image(image, bbox):
    """
    Prepares the image by cutting it in half horizontally and crops the edges to remove serve markers and other stuff
    that could interfere with the text extraction

    :param image: whole input image
    :param bbox: bounding box of the text to extract
    :return: tuple of (image for upper player, image for bottom player)
    """

    y_shrink = 10
    x_shrink = 10
    bbox[0] += x_shrink
    bbox[1] += y_shrink
    bbox[2] -= x_shrink
    bbox[3] -= y_shrink

    bbox = np.rint(bbox).astype(int)
    y_center = (bbox[1] + bbox[3]) / 2
    y_center = np.rint(y_center).astype(int)

    # assumes players are placed on top and bottom half
    image1 = image[bbox[1]:y_center, bbox[0]:bbox[2], ...]
    image2 = image[y_center:bbox[3], bbox[0]:bbox[2], ...]

    cut_image = [image1, image2]
    for i, img in enumerate(cut_image):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, cut_image[i] = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        cut_image[i] = color_normalization(cut_image[i])

    return cut_image[0], cut_image[1]


def color_normalization(image):
    """
    Does the color inversion for the parts of image that have dark text on the bright background. Tesseract performs
    better with dark text on light background.

    :param image: binary input image
    :return: processed binary image
    """
    height, width = np.shape(image)

    # low pass filter
    blurred_image = cv2.GaussianBlur(image, (19, 19), 10)
    # brightness for each vertical strip
    brightness = np.sum(blurred_image, axis=0)

    # calculation of areas where there is mostly bright and inverting them
    threshold = np.max(brightness) / 2
    binary = image > 0
    for w in range(width):
        if brightness[w] > threshold:
            binary[:, w] = np.logical_not(binary[:, w])

    binary = np.logical_not(binary)
    image = binary.astype(np.uint8) * 255

    return image

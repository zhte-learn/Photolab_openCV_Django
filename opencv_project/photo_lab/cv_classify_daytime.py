import cv2
import numpy as np


def classify_daytime(image_path):
    image = cv2.imread(image_path)
    results = {}

    # RGB
    #  grayscale image gives a measure of overall lightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_gray = np.mean(gray)
    threshold_gray = 100
    results["grayscale"] = {
        "mean": mean_gray,
        "threshold": threshold_gray,
        "label": "day" if mean_gray > threshold_gray else "night"
    }

    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # access value channel that represents brightness or intensity
    mean_v = np.mean(hsv[:, :, 2])
    threshold_v = 80
    results["hsv"] = {
        "mean": mean_v,
        "threshold": threshold_v,
        "label": "day" if mean_v > threshold_v else "night"
    }

    # LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # access the L channel that corresponds to lightness
    mean_l = np.mean(lab[:, :, 0])
    threshold_l = 100
    results["lab"] = {
        "mean": mean_l,
        "threshold": threshold_l,
        "label": "day" if mean_l > threshold_l else "night"
    }

    return results

import cv2
import os
import numpy as np
from django.conf import settings


def is_night(
    image_path, 
    brightness_thresh=70,
    dark_pixel_thresh=60,
    dark_ratio_thresh=0.35,
):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]

    mean_brightness = v.mean()
    dark_ratio = np.sum(v < dark_pixel_thresh) / v.size

    if mean_brightness < brightness_thresh or dark_ratio > dark_ratio_thresh:
        return True
    return False


def reduce_noise(image_path):
    image = cv2.imread(image_path)
    smoothed_image = cv2.bilateralFilter(image, 7, 100, 100)
    filename = os.path.basename(image_path)
    smoothed_filename = f"smoothed_{filename}"
    folder = os.path.dirname(image_path)
    smoothed_path = os.path.join(folder, smoothed_filename)
    cv2.imwrite(smoothed_path, smoothed_image)
    relative_folder = os.path.relpath(folder, settings.MEDIA_ROOT)
    return os.path.join(settings.MEDIA_URL, relative_folder, smoothed_filename).replace("\\", "/")


def reduce_blur(image_path):
    image = cv2.imread(image_path)

    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    filename = os.path.basename(image_path)
    sharpened_filename = f"sharpened_{filename}"
    folder = os.path.dirname(image_path)
    sharpened_path = os.path.join(folder, sharpened_filename)
    cv2.imwrite(sharpened_path, sharpened_image)
    relative_folder = os.path.relpath(folder, settings.MEDIA_ROOT)
    return os.path.join(settings.MEDIA_URL, relative_folder, sharpened_filename).replace("\\", "/")

import cv2
import os
import numpy as np
from django.conf import settings


def get_image_path(prefix, image_path, image):
    filename = os.path.basename(image_path)
    new_filename = f"{prefix}_{filename}"
    folder = os.path.dirname(image_path)
    new_path = os.path.join(folder, new_filename)
    cv2.imwrite(new_path, image)
    relative_folder = os.path.relpath(folder, settings.MEDIA_ROOT)
    return os.path.join(settings.MEDIA_URL, relative_folder, new_filename).replace("\\", "/")


def reduce_noise(image_path):
    image = cv2.imread(image_path)
    smoothed_image = cv2.bilateralFilter(image, 7, 100, 100)
    
    return get_image_path("smoothed", image_path, smoothed_image)


def reduce_blur(image_path):
    image = cv2.imread(image_path)

    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return get_image_path("sharpened", image_path, sharpened_image)

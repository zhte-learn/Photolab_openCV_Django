import cv2
import numpy as np
import os
from django.conf import settings


def detect_faces(photo):
    image = cv2.imread(photo)
    folder = os.path.dirname(photo)
    base_name = os.path.basename(photo)

    name, ext = os.path.splitext(base_name)
    new_filename = f"detected_{name}{ext}"
    new_path = os.path.join(folder, new_filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frontal_haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    profile_haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    faces_frontal = frontal_haar.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(50, 50),
    )
    faces_profile = profile_haar.detectMultiScale(
        gray, 
        scaleFactor=1.07, 
        minNeighbors=5,
        minSize=(50, 50),
    )

    faces = list(faces_profile) + list(faces_frontal)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(new_path, image)
    relative_folder = os.path.relpath(folder, settings.MEDIA_ROOT)
    return os.path.join(relative_folder, new_filename).replace("\\", "/")


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


def assess_quality(photo):
    pass
    # image = cv2.imread(photo)
    # return {
    #     "blurred": bool(is_blurry(image)),
    # }

# def remove_overlapping_boxes(faces, overlap_thresh=0.1):
#     if len(faces) == 0:
#         return []

#     boxes = np.array(faces)
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = x1 + boxes[:, 2]
#     y2 = y1 + boxes[:, 3]

#     areas = (x2 - x1) * (y2 - y1)
#     order = areas.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)

#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0, xx2 - xx1)
#         h = np.maximum(0, yy2 - yy1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= overlap_thresh)[0]
#         order = order[inds + 1]

#     return boxes[keep].astype("int")

# def is_blurry(image, threshold=150.0):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return laplacian_var < threshold
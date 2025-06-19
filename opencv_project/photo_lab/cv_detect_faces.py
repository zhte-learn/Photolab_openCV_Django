import cv2
import dlib
import time
import os
import numpy as np
from django.conf import settings


APP_DIR = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.join(APP_DIR, "models", "deploy.prototxt.txt")
model_path = os.path.join(APP_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")


def non_max_suppression_fast(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes).astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")


def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result = {}

    def save_image(suffix, image_to_save):
        folder = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        new_filename = f"detected_{suffix}_{name}{ext}"
        new_path = os.path.join(folder, new_filename)
        cv2.imwrite(new_path, image_to_save)
        return os.path.relpath(new_path, settings.MEDIA_ROOT).replace("\\", "/")

    # 1. Haar
    start = time.time()
    image_haar = image.copy()
    haar_cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    ]
    faces = []
    for cascade in haar_cascades:
        faces.extend(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5))
    final_faces = non_max_suppression_fast(faces)
    for (x, y, w, h) in final_faces:
        cv2.rectangle(image_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)
    result["haar"] = {
        "faces": len(final_faces),
        "time": time.time() - start,
        "path": save_image("haar", image_haar)
    }

    # 2. Dlib HOG
    start = time.time()
    image_dlib = image.copy()
    hog_detector = dlib.get_frontal_face_detector()
    dlib_faces = hog_detector(gray, 1)
    for rect in dlib_faces:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(image_dlib, (x, y), (x + w, y + h), (255, 0, 0), 2)
    result["dlib"] = {
        "faces": len(dlib_faces),
        "time": time.time() - start,
        "path": save_image("dlib", image_dlib)
    }

    # 3. DNN SSD
    start = time.time()
    image_dnn = image.copy()
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(image_dnn, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    result["dnn"] = {
        "faces": len(boxes),
        "time": time.time() - start,
        "path": save_image("dnn", image_dnn)
    }

    return result

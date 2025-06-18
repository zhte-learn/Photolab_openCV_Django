import cv2
import os
import time
import dlib
from django.conf import settings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
proto_path = os.path.join(BASE_DIR, "models", "deploy.prototxt.txt")
model_path = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")


# Define IoU function
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Filter overlapping boxes using IoU
def filter_overlapping_faces(all_faces, iou_threshold=0.3):
    filtered = []
    for box in all_faces:
        if all(compute_iou(box, f) < iou_threshold for f in filtered):
            filtered.append(box)
    return filtered


def detect_haar_faces(image_path):
    start = time.time()
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
	# Load cascades
    cascades = [
		cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
		cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
		cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
	]

	# Detect faces using all cascades
    all_faces = []
    for cascade in cascades:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        all_faces.extend(faces)

	# Filter overlapping detections
    unique_faces = filter_overlapping_faces(all_faces)

    # Draw final rectangles
    for (x, y, w, h) in unique_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    output = save_output_image(image, image_path, "haar")
    return output, time.time() - start


def detect_hog_faces(image_path):
    start = time.time()
    
    # Load dlib's HOG-based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

	# Load and preprocess image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces using HOG + SVM
    faces = hog_face_detector(gray, 1)  # 1 means use image pyramid for better detection

    # Draw rectangles
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    output = save_output_image(image, image_path, "hog")
    return output, time.time() - start


def detect_dnn_faces(image_path):
    start = time.time()
    
	# Load the DNN face detector model (Caffe version)
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    # Draw boxes for each face detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output = save_output_image(image, image_path, "dnn")
    return output, time.time() - start


def save_output_image(image, original_path, suffix):
    folder = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    new_filename = f"detected_{suffix}_{name}{ext}"
    new_path = os.path.join(folder, new_filename)
    cv2.imwrite(new_path, image)
    relative_folder = os.path.relpath(folder, settings.MEDIA_ROOT)
    return os.path.join(relative_folder, new_filename).replace("\\", "/")

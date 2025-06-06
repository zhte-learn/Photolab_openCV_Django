import cv2
import csv
import numpy as np
import os
from uuid import uuid4
import json

from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Photo


def index(request):
    photo_list = Photo.objects.all()
    return render(request, "photo_lab/index.html", {"photo_list": photo_list})

def remove_overlapping_boxes(faces, overlap_thresh=0.1):
    if len(faces) == 0:
        return []

    boxes = np.array(faces)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return boxes[keep].astype("int")


def detect_faces(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id)

    original_path = photo.image.path
    folder = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)

    name, ext = os.path.splitext(base_name)
    base_prefix = name.rsplit("_", 1)[0]

    i = 1
    while True:
        new_filename = f"{base_prefix}_{i}{ext}"
        new_path = os.path.join(folder, new_filename)
        if not os.path.exists(new_path):
            break
        i += 1

    image = cv2.imread(original_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frontal_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

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
    filtered_faces = remove_overlapping_boxes(faces)

    for (x, y, w, h) in filtered_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(new_path, image)
    processed_url = photo.image.url.rsplit("/", 1)[0] + "/" + new_filename

    return render(request, 'photo_lab/result.html', {
        'original': photo.image.url,
        'processed': processed_url,
        'title': photo.title
    })


def annotate_objects(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id)

    original_path = photo.image.path
    folder = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)

    name, ext = os.path.splitext(base_name)
    base_prefix = name.rsplit("_", 1)[0]

    i = 1
    while True:
        new_filename = f"{base_prefix}_{i}{ext}"
        new_path = os.path.join(folder, new_filename)
        if not os.path.exists(new_path):
            break
        i += 1
    
    image = cv2.imread(original_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(new_path, image)
    processed_url = photo.image.url.rsplit("/", 1)[0] + "/" + new_filename

    return render(request, 'photo_lab/annotate.html', {
        'original': photo.image.url,
        'processed': processed_url,
        'title': photo.title
    })


# @csrf_exempt
# def save_annotations(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             image_name = data.get('image_name')
#             annotations = data.get('annotations', [])

#             if not image_name or not annotations:
#                 return JsonResponse({'error': 'Missing data'}, status=400)

#             images_dir = os.path.join(settings.BASE_DIR, 'static', 'img')

#             base_name = os.path.splitext(image_name)[0]
#             save_path = os.path.join(images_dir, f"{base_name}_annotations.csv")

#             with open(save_path, mode='w', newline='', encoding='utf-8') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(['Label', 'X', 'Y', 'Width', 'Height', 'Color'])
#                 for ann in annotations:
#                     writer.writerow([
#                         ann['label'], ann['x'], ann['y'], ann['width'], ann['height'], ann['color']
#                     ])
#             return JsonResponse({'message': 'Annotations saved'}, status=200)
        
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
    
#     return JsonResponse({'error': 'Invalid method'}, status=405)

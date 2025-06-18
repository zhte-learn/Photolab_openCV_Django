import os
from django.conf import settings
from django.shortcuts import render, get_object_or_404
from photo_lab.models import Photo
from .opencv_utils import detect_object


def index(request):
    landmark_photos = Photo.objects.filter(annotation__iexact="muneera")
    return render(request, "landmark_detection/index.html", {"photos": landmark_photos})


def detect_landmark(request, photo_id):
    if request.method == "POST":
        photo = get_object_or_404(Photo, id=photo_id)
        image_path = os.path.join(settings.MEDIA_ROOT, photo.image.name)
        processed_image_url = detect_object(image_path)

    return render(request, "landmark_detection/detection_result.html", {
            "photo": photo,
            "processed_image_url": processed_image_url,
        })

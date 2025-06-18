import os
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from .models import Photo
from .opencv_utils import (
    is_night,
    reduce_noise,
    reduce_blur,
)
from .cv_similarity_retrieval import retrieve_similar_images
from .cv_assess_quality import assess_quality
from .cv_detect_faces import detect_haar_faces, detect_hog_faces, detect_dnn_faces


def index(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        title = image.name
        photo = Photo.objects.create(title=title, image=image)
        return redirect("photo_lab:photo_options", photo_id=photo.id)
    return render(request, "photo_lab/index.html")


def photo_options(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id)
    return render(request, "photo_lab/options.html", {"photo": photo})


def process_photo(request, photo_id):
    photo = get_object_or_404(Photo, id=photo_id)
    operations = request.POST.getlist("operations")
    processed_url = photo.image.url
    time_of_day = None
    quality_assessment = None
    quality_visuals = []
    compare = False
    similar_photos = []
    face_results = {}

    if "detect_faces" in operations:
        haar_path, haar_time = detect_haar_faces(photo.image.path)
        hog_path, hog_time = detect_hog_faces(photo.image.path)
        dnn_path, dnn_time = detect_dnn_faces(photo.image.path)

        face_results = {
            "haar": {"url": settings.MEDIA_URL + haar_path, "time": round(haar_time, 3)},
            "hog": {"url": settings.MEDIA_URL + hog_path, "time": round(hog_time, 3)},
            "dnn": {"url": settings.MEDIA_URL + dnn_path, "time": round(dnn_time, 3)},
        }

    if "detect_daytime" in operations:
        time_of_day = "Night" if is_night(photo.image.path) else "Day"

    if "reduce_noise" in operations:
        processed_url = reduce_noise(photo.image.path)
        compare = True

    if "reduce_blur" in operations:
        processed_url = reduce_blur(photo.image.path)
        compare = True
    
    if "assess_quality" in operations:
        quality_assessment, quality_visuals = assess_quality(photo.image.path)

    if "similarity_retrieval" in operations:
        all_photos = Photo.objects.filter(annotation="sim").exclude(id=photo.id)
        results = retrieve_similar_images(photo.image.path, all_photos, top_k=3)
        
        for score, detail, similar_photo in results:
            similar_photos.append({
                "photo": similar_photo,
                "score": score,
                "details": detail
            })

    return render(request, "photo_lab/result.html", {
        "original": photo.image.url,
        "processed": processed_url,
        "title": photo.title,
        "time_of_day": time_of_day,
        "quality_assessment": quality_assessment,
        "quality_visuals": quality_visuals,
        "photo": photo,
        "compare": compare,
        "similar_photos": similar_photos,
        "face_results": face_results,
    })

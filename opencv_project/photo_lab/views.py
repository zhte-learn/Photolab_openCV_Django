import os
from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from .models import Photo
from .opencv_utils import (
    reduce_noise,
    reduce_blur,
)
from .cv_similarity_retrieval import retrieve_similar_images
from .cv_assess_quality import assess_quality
from .cv_detect_faces import detect_faces
from .cv_classify_daytime import classify_daytime


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
    daytime_results = {}
    quality_assessment = None
    quality_visuals = []
    compare = False
    similar_photos = []
    face_results = {}

    if "detect_faces" in operations:
        detection_result = detect_faces(photo.image.path)

        face_results = {
            "haar": {
                "url": settings.MEDIA_URL + detection_result["haar"]["path"],
                "time": round(detection_result["haar"]["time"], 3),
                "faces": detection_result["haar"]["faces"],
            },
            "dlib": {
                "url": settings.MEDIA_URL + detection_result["dlib"]["path"],
                "time": round(detection_result["dlib"]["time"], 3),
                "faces": detection_result["dlib"]["faces"],
            },
            "dnn": {
                "url": settings.MEDIA_URL + detection_result["dnn"]["path"],
                "time": round(detection_result["dnn"]["time"], 3),
                "faces": detection_result["dnn"]["faces"],
            },
        }

    if "detect_daytime" in operations:
        daytime_results = classify_daytime(photo.image.path)
        labels = [v["label"] for v in daytime_results.values()]
        time_of_day = "Day" if labels.count("day") > labels.count("night") else "Night"

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
        "daytime_results": daytime_results,
        "quality_assessment": quality_assessment,
        "quality_visuals": quality_visuals,
        "photo": photo,
        "compare": compare,
        "similar_photos": similar_photos,
        "face_results": face_results,
    })

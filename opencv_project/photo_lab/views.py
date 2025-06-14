from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from .models import Photo
from .opencv_utils import (
    detect_faces,
    is_night,
    assess_quality,
    reduce_noise,
    reduce_blur,
)


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
    compare = False

    if "detect_faces" in operations:
        relative_url = detect_faces(photo.image.path)
        processed_url = settings.MEDIA_URL + relative_url

    if "detect_daytime" in operations:
        time_of_day = "Night" if is_night(photo.image.path) else "Day"

    if "reduce_noise" in operations:
        processed_url = reduce_noise(photo.image.path)
        compare = True

    if "reduce_blur" in operations:
        processed_url = reduce_blur(photo.image.path)
        compare = True
    
    if "assess_quality" in operations:
        quality_assessment = assess_quality(photo.image.path)

    return render(request, "photo_lab/result.html", {
        "original": photo.image.url,
        "processed": processed_url,
        "title": photo.title,
        "time_of_day": time_of_day,
        # "quality_assessment": quality_assessment,
        "photo": photo,
        "compare": compare,
    })

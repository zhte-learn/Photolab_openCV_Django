import os
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .opencv_utils import detect_object


def index(request):
    return render(request, "landmark_detection/index.html")


def detect_landmark(request):
    if request.method == "POST" and request.FILES.get("image"):
        uploaded_image = request.FILES["image"]
        selected_option = request.POST.get("options")

        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)

        if selected_option == "detect_hq":
            xml_file = os.path.join(settings.BASE_DIR, "cascades", "myhqdetector.xml")
        elif selected_option == "detect_mun":
            xml_file = os.path.join(settings.BASE_DIR, "cascades", "myhousedetector.xml")
        
        processed_image_url = detect_object(image_path, xml_file)

        return render(request, "landmark_detection/detection_result.html", {
            "processed_image_url": processed_image_url,
            "filename": uploaded_image.name,
        })
    
    return render(request, "landmark_detection/detection_result.html", {
        "error": "No image uploaded.",
    })

from django.urls import path

from photo_lab import views


app_name = "photo_lab"

urlpatterns = [
    path("", views.index, name="index"),
    path("detect_faces/<int:photo_id>/", views.detect_faces, name="detect_faces"),
    path("annotate_objects/<int:photo_id>/", views.annotate_objects, name="annotate_objects"),
    # path("save_annotations/", views.save_annotations, name="save_annotations"),
]

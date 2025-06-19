from django.urls import path
from landmark_detection import views


app_name = "landmark_detection"

urlpatterns = [
    path("", views.index, name="index"),
    path("detect_landmark/", views.detect_landmark, name="detect_landmark"),
    # path("detect_landmark/<int:photo_id>/", views.detect_landmark, name="detect_landmark"),
]

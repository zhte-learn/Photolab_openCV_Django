from django.urls import path
from landmark_detection import views


app_name = "landmark_detection"

urlpatterns = [
    path("", views.index, name="index"),
    path("run_detection/<int:photo_id>/", views.run_detection, name="run_detection"),
]

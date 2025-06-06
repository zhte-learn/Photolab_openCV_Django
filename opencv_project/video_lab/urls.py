from django.urls import path

from video_lab import views


app_name = "video_lab"

urlpatterns = [
    path("", views.index, name="index"),
]

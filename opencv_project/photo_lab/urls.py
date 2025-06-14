from django.urls import path
from photo_lab import views


app_name = "photo_lab"

urlpatterns = [
    path("", views.index, name="index"),
    path("photo/<int:photo_id>/", views.photo_options, name="photo_options"),
    path("process_photo/<int:photo_id>/", views.process_photo, name="process_photo"),
]

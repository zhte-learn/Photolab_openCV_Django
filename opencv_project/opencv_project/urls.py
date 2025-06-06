from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from django.urls import path, include


urlpatterns = [
    path('', include('photo_lab.urls', namespace='photo_lab')),
    path('video/', include('video_lab.urls', namespace='video_lab')),
    path('admin/', admin.site.urls),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

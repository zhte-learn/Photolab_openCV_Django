from django.db import models

class Photo(models.Model):
    title = models.CharField("Title", max_length=30)
    image = models.ImageField("Photo", upload_to="photo_images", blank=True)
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Uploaded"
    )

    class Meta:
        verbose_name = "photo"
        verbose_name_plural = "photos"
        ordering = ("-created_at",)

    def __str__(self):
        return self.title[:20] + '...' if len(self.title) > 20 else self.title

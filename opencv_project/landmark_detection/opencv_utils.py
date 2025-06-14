import cv2
import os

from django.conf import settings


def detect_object(image_path):
    # if request.method == "POST":
    #     photo = get_object_or_404(Photo, id=photo_id)
    #     image_path = os.path.join(settings.MEDIA_ROOT, photo.image.name)

    landmark_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, "myhousedetector.xml"))
    image = cv2.imread(image_path)
    
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmark = landmark_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        for (x, y, w, h) in landmark:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # processed_filename = f"processed_{photo.id}_{photo.image.name.split('/')[-1]}"
        # processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
        processed_path = os.path.join(
            settings.MEDIA_ROOT, f"processed_{os.path.basename(image_path)}"
        )
        cv2.imwrite(processed_path, image)
        # processed_image_url = os.path.join(settings.MEDIA_URL, processed_filename)
        return os.path.join(settings.MEDIA_URL, f"processed_{os.path.basename(image_path)}")

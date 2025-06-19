import cv2
import os

from django.conf import settings


def detect_object(image_path, xml_path):
    landmark_cascade = cv2.CascadeClassifier(xml_path)
    image = cv2.imread(image_path)
    
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmark = landmark_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(200, 200))
        for (x, y, w, h) in landmark:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        processed_filename = f"processed_{os.path.basename(image_path)}"
        processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
        cv2.imwrite(processed_path, image)

        return os.path.join(settings.MEDIA_URL, processed_filename)
    
    return None

import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_folder = "faces.jpg"
output_folder = "output_faces"

os.makedirs(output_folder, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png']

for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read {filename}, skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

        print(f"Processed and saved: {filename}")

print("Face detection completed for all images.")
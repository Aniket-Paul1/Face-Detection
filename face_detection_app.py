import cv2
import os
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_from_image():
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
    
def detect_faces_from_webcam():
    cap =cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Real-Time Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    
print("Choose an option:\n1. Detect faces from images\n2. Detect faces from webcam\n3. Exit")
choose = input("Enter 1 or 2 or 3: ")

while choose:
    if choose == "3":
        break
    elif choose == "1":
        detect_faces_from_image()
    elif choose == "2":
        detect_faces_from_webcam()
    else:
        print("Invalid Choose...")
        break
    choose = input("Enter a number")


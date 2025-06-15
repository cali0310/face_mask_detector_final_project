import cv2
import tensorflow as tf
import keras
import numpy as np

# Enable GPU memory growth (for TensorFlow)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(r"C:\Users\dell\PycharmProjects\PythonProject\.venv\Lib\haarcascade_frontalface_default.xml")

# Load trained mask detection model
model = keras.models.load_model(r"C:\Users\dell\PycharmProjects\PythonProject\.venv\Lib\mask_recog.h5")

# Define labels and colors
labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Open webcam
video_capture = cv2.VideoCapture(0)

# Ensure the camera is opened
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, img = video_capture.read()
    if not ret or img is None:
        print("Error: Could not capture frame from the camera.")
        continue  # Skip this iteration if no frame is captured

    img = cv2.flip(img, 1)  # Flip image for mirror effect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection

    # Detect faces
    features = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in features:
        face = img[y:y+h, x:x+w]  # Crop face region
        new_face = cv2.resize(face, (224, 224))  # Resize for model
        normalize = new_face / 255.0  # Normalize pixel values
        resize_face = np.reshape(normalize, (1, 224, 224, 3))

        # Predict mask or no mask
        predict = model.predict(resize_face)
        print("Prediction Output:", predict)  # Debugging print

        if predict[0][0] > predict[0][1]:
            color, text = color_dict[1], "Mask on"
        else:
            color, text = color_dict[0], "No Mask"

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("DETECT FACE", img)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):  # Press 'q' to exit
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
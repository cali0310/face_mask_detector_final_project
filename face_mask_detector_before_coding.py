import cv2
import tensorflow as tf
import keras
import numpy as np

#Constants
HAAR_CASCADE_PATH = r"C:\Users\dell\PycharmProjects\PythonProject\.venv\Lib\haarcascade_frontalface_default.xml"
MODEL_PATH = r"C:\Users\dell\PycharmProjects\PythonProject\.venv\Lib\mask_recog.h5"

LABELS_DICT = {0: 'without_mask', 1: 'with_mask'}
COLOR_DICT = {0: (0, 0, 255), 1: (0, 255, 0)}

#Setup GPU Memory Growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu_device in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_device, True)

#Load Model and Classifier
face_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
mask_detection_model = keras.models.load_model(MODEL_PATH)

#Start Webcam Feed
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

#Main Loop
while True:
    frame_captured, frame = video_capture.read()
    if not frame_captured or frame is None:
        print("Error: Could not capture frame from the camera.")
        continue

    flipped_frame = cv2.flip(frame, 1)
    grayscale_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(
        grayscale_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x_pos, y_pos, width, height) in detected_faces:
        face_region = flipped_frame[y_pos:y_pos+height, x_pos:x_pos+width]
        resized_face = cv2.resize(face_region, (224, 224))
        normalized_face = resized_face / 255.0
        input_face = np.reshape(normalized_face, (1, 224, 224, 3))

        prediction = mask_detection_model.predict(input_face)

        if prediction[0][0] > prediction[0][1]:
            color = COLOR_DICT[1]
            label = "Mask on"
        else:
            color = COLOR_DICT[0]
            label = "No Mask"

        cv2.rectangle(flipped_frame, (x_pos, y_pos), (x_pos + width, y_pos + height), color, 2)
        cv2.putText(
            flipped_frame,
            label,
            (x_pos, y_pos - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Mask Detection", flipped_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#Clean Up
video_capture.release()
cv2.destroyAllWindows()
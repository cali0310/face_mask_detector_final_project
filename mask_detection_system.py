import cv2
import tensorflow as tf
import keras
import numpy as np

class MaskDetectionSystem:
    def __init__(self, model_file_path, haar_cascade_file_path):
        self.face_classifier = cv2.CascadeClassifier(haar_cascade_file_path)
        self.mask_model = keras.models.load_model(model_file_path)
        self.configure_gpu_memory_growth()

    def configure_gpu_memory_growth(self):
        available_gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for gpu_device in available_gpu_devices:
            tf.config.experimental.set_memory_growth(gpu_device, True)

    def preprocess_face_image(self, face_image):
        resized_image = cv2.resize(face_image, (224, 224))
        normalized_image = resized_image / 255.0
        reshaped_image = np.reshape(normalized_image, (1, 224, 224, 3))
        return reshaped_image

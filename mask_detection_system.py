import cv2
import tensorflow as tf
import keras
import numpy as np
from mask_detection_base import MaskDetectionBase

class MaskDetectionSystem(MaskDetectionBase):
    def __init__(self, model_file_path, haar_cascade_file_path):
        self.face_classifier = cv2.CascadeClassifier(haar_cascade_file_path)
        self.mask_model = keras.models.load_model(model_file_path)
        self.configure_gpu_memory_growth()

    def load_model(self, model_path):
        return keras.models.load_model(model_path)

    def predict(self, processed_input):
        return self.mask_model.predict(processed_input)
    
    def configure_gpu_memory_growth(self):
        available_gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for gpu_device in available_gpu_devices:
            tf.config.experimental.set_memory_growth(gpu_device, True)

    def preprocess_face_image(self, face_image):
        resized_image = cv2.resize(face_image, (224, 224))
        normalized_image = resized_image / 255.0
        reshaped_image = np.reshape(normalized_image, (1, 224, 224, 3))
        return reshaped_image
    
    def get_mask_prediction(self, face_image):
        processed_image = self.preprocess_face_image(face_image)
        prediction_scores = self.mask_model.predict(processed_image)
        return prediction_scores

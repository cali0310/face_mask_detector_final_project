import cv2
import tensorflow as tf
import keras
import numpy as np

class MaskDetectionSystem:
    def __init__(self, model_file_path, haar_cascade_file_path):
        self.face_classifier = cv2.CascadeClassifier(haar_cascade_file_path)
        self.mask_model = keras.models.load_model(model_file_path)
        self.configure_gpu_memory_growth()

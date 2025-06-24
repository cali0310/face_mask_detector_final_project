import cv2
import tensorflow as tf
import keras
import numpy as np

haar_cascade_file_path = r"C:\\Users\\dell\\PycharmProjects\\PythonProject\\.venv\\Lib\\haarcascade_frontalface_default.xml"
model_file_path = r"C:\\Users\\dell\\PycharmProjects\\PythonProject\\.venv\\Lib\\mask_recog.h5"

label_text_mapping = {0: 'without_mask', 1: 'with_mask'}
bounding_box_color_mapping = {0: (0, 0, 255), 1: (0, 255, 0)}

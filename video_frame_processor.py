import cv2
from mask_detection_system import MaskDetectionSystem

label_text_mapping = {0: 'without_mask', 1: 'with_mask'}
bounding_box_color_mapping = {0: (0, 0, 255), 1: (0, 255, 0)}

class VideoFrameProcessor:
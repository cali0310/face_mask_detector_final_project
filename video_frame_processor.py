import cv2
from mask_detection_system import MaskDetectionSystem

label_text_mapping = {0: 'without_mask', 1: 'with_mask'}
bounding_box_color_mapping = {0: (0, 0, 255), 1: (0, 255, 0)}

class VideoFrameProcessor:
    def __init__(self, mask_detection_system: MaskDetectionSystem):
        self.mask_detection_system = mask_detection_system

    def process_video_frame(self, video_frame):
        flipped_frame = cv2.flip(video_frame, 1)
        grayscale_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.mask_detection_system.face_classifier.detectMultiScale(
            grayscale_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (face_x, face_y, face_width, face_height) in detected_faces:
            face_region = flipped_frame[face_y:face_y + face_height, face_x:face_x + face_width]
            prediction_scores = self.mask_detection_system.get_mask_prediction(face_region)
            predicted_label = 1 if prediction_scores[0][0] > prediction_scores[0][1] else 0
            label_text = label_text_mapping[predicted_label]
            box_color = bounding_box_color_mapping[predicted_label]

            cv2.rectangle(flipped_frame, (face_x, face_y), (face_x + face_width, face_y + face_height), box_color, 2)
            cv2.putText(
                flipped_frame,
                label_text,
                (face_x, face_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        return flipped_frame
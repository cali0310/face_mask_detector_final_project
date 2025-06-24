import cv2
from video_frame_processor import VideoFrameProcessor

class WebcamVideoStream:
    def __init__(self, video_frame_processor: VideoFrameProcessor):
        self.video_frame_processor = video_frame_processor
        self.webcam_capture_stream = cv2.VideoCapture(0)

    def run_stream(self):
        if not self.webcam_capture_stream.isOpened():
            print("Error: Could not open webcam.")
            return
        
        while True:
            ret, frame = self.webcam_capture_stream.read()
            if not ret or frame is None:
                print("Error: Frame capture failed.")
                continue

            processed_frame = self.video_frame_processor.process_video_frame(frame)
            cv2.imshow("Mask Detection", processed_frame)

            if cv2.waitKey(10) & 0xFF == 27: #ESC key to exit
                break

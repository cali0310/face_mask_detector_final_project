import cv2
from video_frame_processor import VideoFrameProcessor

class WebcamVideoStream:
    def __init__(self, video_frame_processor: VideoFrameProcessor):
        self.video_frame_processor = video_frame_processor
        self.webcam_capture_stream = cv2.VideoCapture(0)
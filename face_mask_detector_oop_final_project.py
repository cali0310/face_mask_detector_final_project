from mask_detection_system import MaskDetectionSystem
from video_frame_processor import VideoFrameProcessor
from webcam_video_stream import WebcamVideoStream

# Constants
haar_cascade_file_path = r"C:\\Users\\dell\\PycharmProjects\\PythonProject\\.venv\\Lib\\haarcascade_frontalface_default.xml"
model_file_path = r"C:\\Users\\dell\\PycharmProjects\\PythonProject\\.venv\\Lib\\mask_recog.h5"

if __name__ == '__main__':
    mask_system = MaskDetectionSystem(model_file_path, haar_cascade_file_path)
    frame_processor = VideoFrameProcessor(mask_system)
    webcam_stream = WebcamVideoStream(frame_processor)
    webcam_stream.run_stream()
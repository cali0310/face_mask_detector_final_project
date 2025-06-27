# Mask Detection System

A real-time face mask detection system that uses my webcam to detect people wearing masks or not. It uses OpenCV for face detection and a TensorFlow/Keras model I trained to classify if a detected face has a mask.

## How It Works

- First, it detects faces in the webcam video using a Haar Cascade classifier.
- Then, it runs each face through my mask recognition model.
- The app draws colored boxes around faces and labels them either **with mask** or **without mask**.
- I separated the code into different parts: one for detection, one for processing video frames, and one for managing the webcam stream.

## Files I Created

- `mask_detection_system.py` — handles loading the model and making predictions on faces.  
- `video_frame_processor.py` — processes each webcam frame and annotates faces with mask info.  
- `webcam_video_stream.py` — captures video from the webcam and shows the annotated video.  
- `face_mask_detector_oop_final_project.py` — puts everything together and runs the app.


## What You Need to Run It

- Python 3  
- Libraries: OpenCV, TensorFlow, Keras, Numpy  
- The Haar Cascade XML file and mask model `.h5` file (I put my local paths in the code, update them if you want to try it yourself)

Install dependencies with:

```bash
pip install opencv-python tensorflow keras numpy

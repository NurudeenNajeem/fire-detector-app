import streamlit as st # This is fine, just imports the library
import cv2
import math
from ultralytics import YOLO
import cvzone
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# --- Configuration ---
MODEL_PATH = 'fire.pt'  # Make sure this file is in the same directory as app.py
CLASS_NAMES = ['fire']
DEFAULT_CONFIDENCE_THRESHOLD = 0.3 # Reduced for better initial visibility
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# --- Load YOLO Model (cached for efficiency) ---
@st.cache_resource # <--- THIS IS THE CULPRIT!
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

model = load_yolo_model(MODEL_PATH) # <--- This line calls the cached function, which uses `st.cache_resource`

# --- Streamlit UI ---
#st.set_page_config(page_title="Fire Detection App", layout="wide") # <--- THIS IS WHERE THE ERROR OCCURS
st.title("🔥 Real-time Fire Detection  🔥")
# ... rest of your code

st.sidebar.header("Configuration")
source_option = st.sidebar.radio("Select Video Source:", ("Upload Video", "Webcam"))
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, int(DEFAULT_CONFIDENCE_THRESHOLD * 100)) / 100.0

# --- Video Processing Function ---
def process_frame(frame, model, confidence_thresh, class_names):
    # Resize frame
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Perform inference
    results = model(frame, stream=True)

    # Draw detections
    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence_percent = math.ceil(confidence * 100)
            Class = int(box.cls[0])

            if confidence >= confidence_thresh:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5) # Red bounding box
                # Ensure the text position is within bounds
                text_x = x1 + 8
                text_y = y1 + 30 # Adjusted for better visibility, was 100
                if text_y > frame.shape[0] - 10: # If text would go off screen bottom
                    text_y = frame.shape[0] - 10
                if text_x > frame.shape[1] - 10: # If text would go off screen right
                    text_x = frame.shape[1] - 10

                cvzone.putTextRect(frame, f'{class_names[Class]} {confidence_percent}%', 
                                  [text_x, text_y], scale=1.5, thickness=2)
    return frame

# --- Video Upload Logic ---
if source_option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file (.mp4, .avi, etc.)", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        st.video(uploaded_file) # Show the original uploaded video
        st.subheader("Processing Video for Detections...")

        # Create a temporary file to save the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file for processing.")
        else:
            frame_placeholder = st.empty() # Placeholder for processed frames
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame += 1
                progress_bar.progress(min(100, int((current_frame / total_frames) * 100)))

                processed_frame = process_frame(frame, model, confidence_threshold, CLASS_NAMES)
                frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

            cap.release()
            tfile.close() # Close and delete the temporary file
            st.success("Video processing complete!")

# --- Webcam Logic (using streamlit-webrtc) ---
elif source_option == "Webcam":
    st.info("Ensure your webcam is connected and allowed by the browser.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self, model, confidence_thresh, class_names):
            self.model = model
            self.confidence_thresh = confidence_thresh
            self.class_names = class_names

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            processed_image = process_frame(image, self.model, self.confidence_thresh, self.class_names)
            return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

    webrtc_streamer(
        key="fire-detection-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_transformer_factory=lambda: VideoProcessor(model, confidence_threshold, CLASS_NAMES),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True, # Allows for asynchronous processing, better for models
    )
    st.write("Click 'START' to begin webcam processing.")
    #streamlit run app.py
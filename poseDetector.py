import cv2
import numpy as np
import mediapipe as mp
import joblib
import itertools
import streamlit as st
import tempfile

# Landmark processing functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark[:25]]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    processed_list = [(x - base_x, y - base_y) for x, y in landmark_list]
    processed_list = list(itertools.chain.from_iterable(processed_list))
    max_value = max(map(abs, processed_list))
    return [x / max_value for x in processed_list]

# Bounding box functions
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark[11:25]])
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect, rect_color):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

def draw_info_text(image, brect, label_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    cv2.putText(image, label_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def load_model():
    return joblib.load('pose_XGB_model.pkl')

def process_pose(uploaded_video):
    st.header("Pose Process")
    # Initialize counts
    crossed = raised = explain = straight = face = 0

    # Create a placeholder for displaying frames
    stframe = st.empty()

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Load the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Load Mediapipe Pose model and XGBoost classifier
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    xg_boost_model = load_model()

    # Load classifier labels
    with open('pose_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = f.read().splitlines()

    # Get video FPS and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_time = frame_count / fps

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        debug_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(debug_image)

        if results.pose_landmarks:
            # Calculate bounding rect and landmarks
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            landmark_list = calc_landmark_list(debug_image, results.pose_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            # Predict pose
            facial_emotion_id = xg_boost_model.predict([pre_processed_landmark_list])[0]

            # Update counts based on the predicted pose
            if facial_emotion_id == 0:
                crossed += 1
            elif facial_emotion_id == 1:
                raised += 1
            elif facial_emotion_id == 2:
                explain += 1
            elif facial_emotion_id == 3:
                straight += 1
            elif facial_emotion_id == 4:
                face += 1

            # Draw bounding rect and info text
            rect_color = (0, 255, 0) if facial_emotion_id in [2, 3] else (0, 0, 255)
            draw_bounding_rect(debug_image, brect, rect_color)
            draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])

        # Display the frame in the Streamlit app
        stframe.image(debug_image, channels="RGB")

    # Release the video object
    cap.release()

    # Calculate the time durations for each pose
    crossed_time = crossed / fps if fps > 0 else 0
    raised_time = raised / fps if fps > 0 else 0
    explain_time = explain / fps if fps > 0 else 0
    straight_time = straight / fps if fps > 0 else 0

    # Prepare the results dictionary
    results = {
        "total_video_time": total_video_time,
        "crossed_time": crossed_time,
        "raised_time": raised_time,
        "explain_time": explain_time,
        "straight_time": straight_time,
    }

    # Display the results
    st.write(f"Total video duration: {results['total_video_time']:.2f} seconds")
    st.write(f"Crossed arms time: {results['crossed_time']:.2f} seconds")
    st.write(f"Raised arms time: {results['raised_time']:.2f} seconds")
    st.write(f"Explanation gesture time: {results['explain_time']:.2f} seconds")
    #st.write(f"Straight pose time: {results['straight_time']:.2f} seconds")

    # Return the results
    return total_video_time, crossed_time, raised_time, explain_time


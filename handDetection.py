import cv2
import mediapipe as mp
import time
import streamlit as st
import tempfile

def hand_detection(uploaded_video):
    st.header("Hand Detection")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    hand_visible_frame_count = 0
    total_frame_count = 0

    stframe = st.empty()

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_time = frame_count / fps

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            total_frame_count += 1

            # Convert the image to RGB for Mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                # Increment frame count for visible hands
                hand_visible_frame_count += 1

                # Draw hand landmarks on the image
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame in the Streamlit app
            stframe.image(image, channels="BGR")

        cap.release()

        hand_visible_time = hand_visible_frame_count / fps  # Convert frame count to time in seconds
        hand_visible_percentage = (hand_visible_time / total_video_time) * 100  # Calculate percentage

        st.write(f"Total video duration: {total_video_time:.2f} seconds")
        st.write(f"Total hand visible time: {hand_visible_time:.2f} seconds")
        st.write(f"Percentage of time hands were visible: {hand_visible_percentage:.2f}%")
        
        return total_video_time, hand_visible_time, hand_visible_percentage


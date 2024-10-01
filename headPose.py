import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# Initialize MediaPipe models for face mesh and hand tracking
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)

# Function to check if hand is touching face
def is_hand_touching_face(hand_landmarks, face_landmarks, img_w, img_h):
    face_points_of_interest = [1, 33, 263, 61, 291]  # Nose tip, eyes, mouth corners

    for hand_landmark in hand_landmarks.landmark:
        hand_x, hand_y = hand_landmark.x * img_w, hand_landmark.y * img_h

        for face_landmark in face_points_of_interest:
            face_x = face_landmarks.landmark[face_landmark].x * img_w
            face_y = face_landmarks.landmark[face_landmark].y * img_h

            distance = np.sqrt((hand_x - face_x) ** 2 + (hand_y - face_y) ** 2)

            if distance < 30:
                return True
    return False

# Function to process video for head pose and hand proximity
def head_pose(uploaded_video):
    st.header("Head Pose and Hand Proximity Analysis")

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        video = cv2.VideoCapture(tfile.name)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_time = frame_count / fps

        stframe = st.empty()

        total_frames = 0
        looking_forward_frames = 0
        face_not_center_frames = 0
        hand_touching_face_frames = 0
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            total_frames += 1

            # Flip the frame for selfie-view and convert to RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = frame.shape

            # Process face and hand landmarks
            face_results = face_mesh.process(frame)
            hand_results = hands.process(frame)

            # Convert back to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw grid for centering feedback
            grid_center_x = img_w // 2
            grid_center_y = img_h // 2
            num_lines = 5
            for i in range(1, num_lines):
                cv2.line(frame, (i * img_w // num_lines, 0), (i * img_w // num_lines, img_h), (255, 255, 255), 1)
                cv2.line(frame, (0, i * img_h // num_lines), (img_w, i * img_h // num_lines), (255, 255, 255), 1)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec
                    )

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            if is_hand_touching_face(hand_landmarks, face_landmarks, img_w, img_h):
                                hand_touching_face_frames += 1
                                cv2.putText(frame, "Hand Touching Face", (20, img_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    face_2d = []
                    face_3d = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rotation_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    x_angle, y_angle, z_angle = angles[0] * 360, angles[1] * 360, angles[2] * 360

                    # Determine head pose direction
                    if y_angle < -10:
                        text = "Looking Left"
                    elif y_angle > 10:
                        text = "Looking Right"
                    elif x_angle < -10:
                        text = "Looking Down"
                    elif x_angle > 10:
                        text = "Looking Up"
                    else:
                        text = "Looking Forward"
                        looking_forward_frames += 1

                    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    # Centering Feedback
                    if nose_2d[0] < grid_center_x - 50 or nose_2d[0] > grid_center_x + 50 or nose_2d[1] < grid_center_y - 50 or nose_2d[1] > grid_center_y + 50:
                        face_not_center_frames += 1

            # Display the frame
            stframe.image(frame, channels="BGR")

        video.release()

        # Calculate percentages
        looking_forward_percentage = (looking_forward_frames / total_frames) * 100
        face_not_center_percentage = (face_not_center_frames / total_frames) * 100
        hand_touching_face_percentage = (hand_touching_face_frames / total_frames) * 100

        # Display results
        st.write(f"Total video duration: {total_video_time:.2f} seconds")
        st.write(f"Percentage of time looking forward: {looking_forward_percentage:.2f}%")
        st.write(f"Percentage of time face not in center: {face_not_center_percentage:.2f}%")
        st.write(f"Percentage of time hand touching face: {hand_touching_face_percentage:.2f}%")
    
        return total_video_time, looking_forward_percentage, face_not_center_percentage, hand_touching_face_percentage


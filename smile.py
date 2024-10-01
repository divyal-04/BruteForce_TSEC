import cv2
import streamlit as st
import tempfile

def smile_detection(uploaded_video):
    st.header("Smile Processing")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    smile_frame_count = 0
    total_frame_count = 0

    stframe = st.empty()

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        video = cv2.VideoCapture(tfile.name)
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_time = frame_count / fps

        while video.isOpened():
            check, frame = video.read()
            if not check:
                break

            total_frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

                if len(smiles) > 0:
                    smile_frame_count += 1  
                    for (sx, sy, sw, sh) in smiles:
                        if (x <= sx <= x + w) and (y <= sy <= y + h):
                            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 3)

            stframe.image(frame, channels="BGR")

        video.release()

        smile_time = smile_frame_count / fps
        smile_percentage = (smile_time / total_video_time) * 100

        st.write(f"Total video duration: {total_video_time:.2f} seconds")
        st.write(f"Total smile time: {smile_time:.2f} seconds")
        st.write(f"Percentage of time smiling: {smile_percentage:.2f}%")

        return total_video_time, smile_time, smile_percentage
import streamlit as st
import deepspeech
import numpy as np
import io
import pydub
import speech_recognition as sr

# Load DeepSpeech model
model_file_path = 'C:\\Users\\Hp\\Downloads\\BruteForce_TSEC\\deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'C:\\Users\\Hp\\Downloads\\BruteForce_TSEC\\deepspeech-0.9.3-models.scorer'

model = deepspeech.Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

# File path for saving results
output_file_path = "C:\\Users\\Hp\\Downloads\\BruteForce_TSEC\\resume_processing\\answers.txt"

# Function to save results to a file
def save_results(transcript, wpm, clarity, snr, duration):
    with open(output_file_path, 'w') as f:
        f.write("Transcript:\n")
        f.write(f"{transcript}\n\n")
        f.write(f"Speech Duration: {duration:.2f} seconds\n")
        f.write(f"Words Per Minute: {wpm:.2f}\n")
        f.write(f"Clarity Score (0 to 1): {clarity:.2f}\n")
        f.write(f"SNR (dB): {snr:.2f} dB\n")

# Function to convert audio to text using SpeechRecognition
def speech_to_text(audio_data, file_extension):
    try:
        if file_extension == "mp3":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        elif file_extension == "wav":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
        else:
            return ""

        wav_io = io.BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return ""

# Function to calculate words per minute
def calculate_wpm(audio_data, transcript, file_extension):
    try:
        if file_extension == "mp3":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        elif file_extension == "wav":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data))

        duration_in_sec = len(sound) / 1000.0
        word_count = len(transcript.split())
        wpm = (word_count / duration_in_sec) * 60

        return wpm, duration_in_sec
    except Exception as e:
        return None, None

# Function to calculate clarity based on audio energy and SNR
def calculate_clarity(audio_data, file_extension):
    try:
        if file_extension == "mp3":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        elif file_extension == "wav":
            sound = pydub.AudioSegment.from_file(io.BytesIO(audio_data))

        # Convert to numpy array
        samples = np.array(sound.get_array_of_samples())
        if sound.channels == 2:  # Handle stereo audio
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # RMS Energy
        rms_energy = np.sqrt(np.mean(samples**2))

        # Noise estimation using a simple method (first few seconds)
        noise_samples = samples[:int(0.1 * len(samples))]  # First 0.1 seconds
        noise_rms = np.sqrt(np.mean(noise_samples**2))

        # SNR Calculation
        if noise_rms > 0:
            snr = 20 * np.log10(rms_energy / noise_rms)
        else:
            snr = float('inf')  # Infinite SNR if noise is zero

        # Normalize clarity score
        clarity = (rms_energy / (rms_energy + noise_rms)) if (rms_energy + noise_rms) > 0 else 0

        return clarity, snr
    except Exception as e:
        return None, None

# Function to process audio file
def process_audio(file_path):
    try:
        st.write("Opening the file...")
        audio_file = open(file_path, 'rb')
        file_extension = file_path.split('.')[-1].lower()
        audio_bytes = audio_file.read()
        audio_file.close()

        st.write(f"File extension: {file_extension}")
        st.write(f"File size: {len(audio_bytes)} bytes")

        # Show processing message
        st.write("Processing video... Please wait.")

        # Convert speech to text using SpeechRecognition
        transcript = speech_to_text(audio_bytes, file_extension)

        if transcript:
            st.write(f"Transcript generated: {transcript[:100]}")  # Check the first 100 characters
            # Calculate WPM
            wpm, duration_in_sec = calculate_wpm(audio_bytes, transcript, file_extension)

            if wpm is not None:
                st.write(f"WPM: {wpm}, Duration: {duration_in_sec}")
                # Calculate Clarity
                clarity, snr = calculate_clarity(audio_bytes, file_extension)

                if clarity is not None:
                    st.write(f"Clarity: {clarity}, SNR: {snr}")
                    # Save results to file
                    save_results(transcript, wpm, clarity, snr, duration_in_sec)
                    st.success("For further analysis check Communication Skills Tab!")
                else:
                    st.error("There was an issue calculating Clarity.")
            else:
                st.error("There was an issue calculating WPM.")
        else:
            st.error("There was an issue processing the file.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
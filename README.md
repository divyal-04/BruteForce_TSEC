# InterviewIQ

## Introduction
The **InterviewIQ** website helps users improve their interview performance by practicing various skills such as behavioral interview questions, communication, and teamwork. The app uses AI to analyze videos and audio, providing real-time feedback. It also generates resume-related questions and provides material to help users succeed in interviews.

---

## Features
- **Behavioral Questions**: Analyze a video of your mock interview to receive feedback on non-verbal communication (e.g., eye contact, hand gestures, posture).
- **Communication Skills**: Upload a video and get feedback on verbal communication like tone, pacing, and clarity.
- **Resume-based Questions**: Automatically generate and practice answering questions related to your resume.
- **Materials**: Practice aptitude questions, receive career course suggestions, and access learning roadmaps.
- **Teamwork & Collaboration**: Test your teamwork and collaboration skills with simulated scenarios.

---

## Installation Guide

### Prerequisites
- Python 3.9
- Google Generative AI API Key
- Streamlit
- Required Python libraries (listed in the `requirements.txt`)

### Steps to Install

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/divyal-04/BruteForce_TSEC.git
    ```

2. **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**:
   Create a `.env` file and add your **Google API Key**:
      ```
      GOOGLE_API_KEY=your-google-api-key
      ```

4. **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## Usage Instructions

### Behavioral Questions Analysis
The **Behavioral Questions** section helps analyze your body language and non-verbal communication. The analysis is based on your video recording. The features in this section are powered by the following modules:

- **headPose.py**: Detects head positioning and gives feedback on whether you’re maintaining eye contact and facing forward.
- **poseDetector.py**: Analyzes body movements like crossed arms, raised hands, or gestures that might indicate confidence, discomfort, or engagement.
- **handDetection.py**: Detects hand visibility and gestures (e.g., hands in view, touching the face).
- **smile.py**: Evaluates whether you are smiling and assesses the duration of your smile during the interview.

**Steps to use**:
1. Select **Behavioral Questions** from the sidebar.
2. Upload a video file (MP4, MOV, or AVI format) of your mock interview.
3. The app will analyze the video using the mentioned modules, generating insights such as:
   - Time spent looking forward.
   - Percentage of time with the face not centered.
   - Hand gestures, smile detection, and overall engagement.
4. The system will display a report and provide a downloadable PDF with feedback.

### Communication Skills Analysis
In the **Communication Skills** section, this evaluates your verbal communication in mock interview videos, including aspects like tone, clarity, and pacing. The following modules are involved:

- **resume_results.py**: This file processes your video to analyze your verbal responses, checking clarity and appropriateness of your answers based on your resume content.
- **analyze_video**: This module is used to process your video and generate insights on your spoken communication (tone, volume, clarity).

**Steps to use**:
1. Select **Communication Skills** from the sidebar.
2. Click **Analyze Audio** to process your video.
3. It will process your audio, evaluate communication aspects, and give you feedback on areas to improve.

### Materials Section
The **Materials** section provides additional resources for interview preparation:

- **Aptitude**: Includes a set of aptitude questions for you to practice.
- **Suggest Courses**: The app suggests career-related courses and educational resources to enhance your skills.

Modules involved:
- **aptitude.py**: Responsible for generating and displaying aptitude questions.
- **suggestCourses.py**: Suggests courses based on your career goals and skill requirements.

### Teamwork & Collaboration
In the **Teamwork and Collaboration** section, you can practice your teamwork skills with scenarios provided by the app. The following module is used to simulate these scenarios:

- **collab.py**: Contains the logic to simulate teamwork and collaboration scenarios for testing how well you handle situations that require group interaction. Mimics real life case studies and scenarios as well.

### Graphical Outputs & PDF Reports
After the video analysis, the app generates various graphs to visualize your performance. These graphs are created using **Matplotlib** and **Seaborn**. They include:

- **Behavioral Graphs**: A comprehensive chart showing how you performed in different categories like hand visibility, smile, body language, and overall score.
- **Time Series Graphs**: Displays how your behavior changed over time during the video.
- **Heatmap**: A heatmap showing the intensity of your engagement over time based on factors like eye contact and hand gestures.

The app compiles these graphs into a **PDF report** that you can download.

---

## Technologies Used
- **Streamlit**: Web framework for building the user interface.
- **Google Generative AI**: AI responses for feedback on communication.
- **OpenCV** and **Dlib**: For facial and hand gesture detection.
- **Matplotlib** and **Seaborn**: For creating behavioral analysis graphs.
- **LangChain**: For question answering using Google Generative AI.
- **ReportLab**: For generating and formatting the PDF reports.

---

## Project Structure
Here is a quick overview of the project files:

```bash
.
├── app.py                      # Main Streamlit app file
├── .env                        # Environment variables for Google API
├── resumeQuestions.py           # Resume-based questions generator
├── smile.py                     # Smile detection module
├── handDetection.py             # Hand gesture detection module
├── poseDetector.py              # Pose detection for body language
├── headPose.py                  # Head pose detection
├── collab.py                    # Teamwork and collaboration scenarios
├── aptitude.py                  # Aptitude question generator
├── suggestCourses.py            # Suggest courses based on career goals
├── resume_results.py            # Analyze video for resume-based answers
├── haarcascade_frontalface_default.xml  # Facial recognition model
├── haarcascade_smile.xml        # Smile detection model
└── requirements.txt             # Python dependencies

---


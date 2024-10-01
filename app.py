import streamlit as st
from resumeQuestions import run_resume_interview_generator
from smile import smile_detection 
from handDetection import hand_detection
from poseDetector import process_pose
from headPose import head_pose
from collab import teamwork_collaboration_scenario
from aptitude import aptitude_question_generator
from suggestCourses import gemini_suggest_courses
from resume_results import analyze_video

import google.generativeai as genai
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain.schema import Document
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tempfile

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set the title of the app
st.title("Mock Interview Practice")

# Create a sidebar with options
st.sidebar.title("Interview Categories")
options = st.sidebar.radio("Select a category:", 
                            ("Behavioral Questions", 
                             "Communication Skills", 
                             "Resume-based Questions", 
                             "Materials",
                             "Teamwork and Collaboration"))

# Behavioral Questions Section
if options == "Behavioral Questions":
    st.header("Behavioral Questions")
    st.write("Upload a video of your practice session:")
    
    video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    # Initialize variables
    total_video_time = 0
    looking_forward_percentage = 0
    face_not_center_percentage = 0
    hand_touching_face_percentage = 0
    crossed_time = 0
    raised_time = 0
    explain_time = 0
    hand_visible_time = 0
    hand_visible_percentage = 0
    smile_time = 0
    smile_percentage = 0
    time_not_engaged = 0
    overall_non_verbal_score = 0
    time_series_data ={}
    
    if video_file is not None:
        total_video_time, looking_forward_percentage, face_not_center_percentage, hand_touching_face_percentage = head_pose(video_file)
        video_file.seek(0)
        total_video_time, crossed_time, raised_time, explain_time = process_pose(video_file)
        video_file.seek(0)
        total_video_time, hand_visible_time, hand_visible_percentage = hand_detection(video_file)
        video_file.seek(0)        
        total_video_time, smile_time, smile_percentage = smile_detection(video_file)       

    # Additional metrics
        time_not_engaged = total_video_time - (total_video_time * (looking_forward_percentage / 100))
        overall_non_verbal_score = (looking_forward_percentage + (100 - face_not_center_percentage) + (100 - hand_touching_face_percentage) + 
                                    (100 - (crossed_time / total_video_time * 100)) + (raised_time / total_video_time * 100) + 
                                    (hand_visible_percentage) + (smile_percentage)) / 7  # Average score from 7 metrics

        # Generate time series data (simulated)
        total_video_time = int(total_video_time)  # or use round(total_video_time)
        time_series_data = {
            "Time": np.arange(1, total_video_time + 1),
            "Looking Forward": np.random.randint(0, 100, total_video_time),
            "Hand Visible": np.random.randint(0, 100, total_video_time),
        }

        # Simulate engagement data for heatmap
        engagement_data = np.random.rand(total_video_time, 2) * 100  # Random intensity values

    def get_conversational_chain():
        prompt_template = """
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        return chain

    context = f"""
    You are analyzing a video recording of an interview preparation session. The following behavioral analysis has been conducted:

    - Total video duration: {total_video_time} seconds
    - Percentage of time looking forward: {looking_forward_percentage}%
    - Percentage of time face not in center: {face_not_center_percentage}%
    - Percentage of time hand touching face: {hand_touching_face_percentage}%
    - Crossed arms time: {crossed_time} seconds
    - Raised arms time: {raised_time} seconds
    - Explanation gesture time: {explain_time} seconds
    - Hand visible time: {hand_visible_time} seconds
    - Hand visible percentage: {hand_visible_percentage}%
    - Smile time: {smile_time} seconds
    - Smile percentage: {smile_percentage}%
    - Percentage of time not engaged: {time_not_engaged}
    - Overall non-verbal communication score: {overall_non_verbal_score}

    Provide an in-depth non-verbal body language analysis. It should cover everything and more.
    Elaborate how too much or too little of those actions may be positive or negative.
    """

    documents = [Document(page_content=context)]

    question = "Based on this analysis, what tips, feedback, compliments can you provide to assess and analyze the candidate's non-verbal communication skills during interviews based on the values provided? Give elaborate feedback."

    chain = get_conversational_chain()

    answer = chain.run(input_documents=documents, question=question)

    # Create a graph using matplotlib
    def create_behavioral_graph():
        labels = ['Looking Forward', 'Face Not Center', 'Hand Touching Face', 'Crossed Arms', 
                'Raised Arms', 'Explanation Gesture', 'Hand Visible', 'Smile', 'Not Engaged', 'Overall Score']
        values = [
            looking_forward_percentage, 
            face_not_center_percentage, 
            hand_touching_face_percentage, 
            crossed_time, 
            raised_time, 
            explain_time, 
            hand_visible_percentage, 
            smile_percentage, 
            time_not_engaged, 
            overall_non_verbal_score
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, color='lightblue')
        ax.set_ylabel('Percentage / Time (seconds)')
        ax.set_title('Comprehensive Behavioral Analysis during Interview Preparation')
        plt.xticks(rotation=45, ha='right')

        # Save the figure to a temporary file
        img_path = tempfile.mktemp(suffix=".png")
        plt.savefig(img_path, format='png', bbox_inches='tight')  # Use bbox_inches to fit the image tightly
        plt.close(fig)
        return img_path

    def create_time_series_graph():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_series_data["Time"], time_series_data["Looking Forward"], label='Looking Forward', marker='o')
        ax.plot(time_series_data["Time"], time_series_data["Hand Visible"], label='Hand Visible', marker='o')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Time Series Analysis of Non-Verbal Behaviors')
        ax.legend()
        
        img_path = tempfile.mktemp(suffix=".png")
        plt.savefig(img_path, format='png', bbox_inches='tight')
        plt.close(fig)
        return img_path

    def create_heatmap_engagement_intensity():
        plt.figure(figsize=(10, 6))
        sns.heatmap(engagement_data, cmap="YlGnBu", cbar=True)
        plt.title('Engagement Intensity Over Time')
        plt.xlabel('Behavior Types')
        plt.ylabel('Time (seconds)')
        plt.xticks(ticks=[0.5, 1.5], labels=['Looking Forward', 'Hand Visible'], rotation=0)
        plt.yticks(np.arange(total_video_time), np.arange(1, total_video_time + 1), rotation=0)

        img_path = tempfile.mktemp(suffix=".png")
        plt.savefig(img_path, format='png', bbox_inches='tight')
        plt.close()
        return img_path

    def create_engagement_bar_chart():
        labels = ['Engaged', 'Not Engaged']
        values = [total_video_time * (looking_forward_percentage / 100), time_not_engaged]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, color=['lightgreen', 'salmon'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Percentage of Time Engaged vs. Not Engaged')
        
        img_path = tempfile.mktemp(suffix=".png")
        plt.savefig(img_path, format='png', bbox_inches='tight')
        plt.close(fig)
        return img_path

    # Create a function to generate the PDF
    def create_pdf(answer, graph_paths):
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter

        # Define margins
        margin_left = 50
        margin_top = 50
        margin_right = 50
        margin_bottom = 50
        available_width = width - margin_left - margin_right

        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin_left, height - margin_top, "Interview Preparation Analysis Report")

        # Add the analysis answer
        c.setFont("Helvetica", 12)
        text = c.beginText(margin_left, height - margin_top - 20)

        # Split the answer into lines and handle text wrapping
        for line in answer.split('\n'):
            words = line.split(' ')
            current_line = ""

            for word in words:
                test_line = current_line + (word + " ") if current_line else word + " "
                text_width = c.stringWidth(test_line)

                if text_width > available_width:
                    text.textLine(current_line)
                    current_line = word + " "

                    if text.getY() <= margin_bottom + 50:
                        c.drawText(text)
                        c.showPage()
                        c.setFont("Helvetica", 12)
                        text = c.beginText(margin_left, height - margin_top)
                        text.setTextOrigin(margin_left, height - margin_top)

                else:
                    current_line = test_line  

            if current_line:
                text.textLine(current_line)

        c.drawText(text)

        # Draw each graph on a new page
        for graph_path in graph_paths:
            c.showPage()
            c.drawImage(graph_path, 72, height - 350, width=width - 144, height=300)

        c.showPage()
        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    # Create the graphs and get the image paths
    graph_paths = [
        create_behavioral_graph(),
        create_time_series_graph(),
        create_heatmap_engagement_intensity(),
        create_engagement_bar_chart()
    ]

    # Create the PDF
    pdf_buffer = create_pdf(answer, graph_paths)

    # Provide download link for the PDF
    st.download_button("Download PDF Report", data=pdf_buffer, file_name="interview_analysis_report.pdf", mime="application/pdf")

# Communication Skills Section
elif options == "Communication Skills":
    st.header("Communication Skills")
    st.write("Click the button below to analyze the audio.")
    
    if st.button("Analyze Audio"):
        analyze_video()  # Directly call the analyze_video() function here
        st.write("Video processed.")


# Resume-based Questions Section
elif options == "Resume-based Questions":
    run_resume_interview_generator()



# Materials Section
elif options == "Materials":
    def display_aptitude_materials():
        aptitude_question_generator()
        
    def suggest_courses():
        gemini_suggest_courses()
        

    tab1, tab2, tab3 = st.tabs(["Aptitude", "Suggest Course", "Roadmap"])
    with tab1:
        display_aptitude_materials()

    with tab2:
        suggest_courses()
        

# Teamwork Section
elif options == "Teamwork and Collaboration":
    teamwork_collaboration_scenario()



# Footer
st.sidebar.markdown("### Get Ready for Success!")
st.sidebar.markdown("Practice makes perfect. Good luck with your interview!")

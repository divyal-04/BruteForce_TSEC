import streamlit as st
import os
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from moviepy.editor import VideoFileClip  # Import MoviePy for audio extraction
from transcript_processing import process_audio  # Adjusted import

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the Document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a professional interviewer. Based on the resume content provided below, generate relevant interview questions that can help assess the candidate's skills, experience, and suitability for a job role.
    Give a maximum of 5 questions in total from different categories.
    Do not give them number indexing.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def save_questions_to_file(questions, file_path):
    """Save generated questions to a specified text file."""
    with open(file_path, 'w') as f:  # Open file in write mode to truncate it
        for question in questions:
            f.write(f"{question}\n")  # Write each question on a new line

def run_resume_interview_generator():
    """Run the Resume Interview Question Generator interface."""
    st.header("📝 Resume Interview Question Generator")
    st.write("Upload your resume in PDF format, and generate relevant interview questions!")

    # File uploader for resume
    pdf_docs = st.file_uploader("Upload Your Resume (PDF)", type="pdf", accept_multiple_files=False)

    if pdf_docs:
        if st.button("Process Resume"):
            with st.spinner("Processing..."):
                # Extract text and generate vector store
                processed_pdf_text = get_pdf_text([pdf_docs])
                text_chunks = get_text_chunks(processed_pdf_text)
                get_vector_store(text_chunks)
                st.success("Resume processed successfully!")

                # Generate interview questions
                chain = get_conversational_chain()
                documents = [Document(chunk) for chunk in text_chunks]
                questions_response = chain({"input_documents": documents, "question": "Generate interview questions."}, return_only_outputs=True)

                # Display generated questions
                st.subheader("🎉 Generated Interview Questions:")
                questions = questions_response["output_text"].split('\n')
                question_list = [question.strip() for question in questions if question.strip()]

                if question_list:
                    for i, question in enumerate(question_list):
                        st.write(f"{i + 1}. {question}")

                # Store questions in session state for reuse
                if 'questions' not in st.session_state:
                    st.session_state.questions = question_list

                # Save questions to the specified file
                questions_file_path = r"C:\Users\Hp\Downloads\BruteForce_TSEC\resume_processing\questions.txt"
                save_questions_to_file(question_list, questions_file_path)

                # Button to generate more questions
                if st.button("Generate More Questions"):
                    with st.spinner("Generating more questions..."):
                        additional_questions_response = chain({"input_documents": documents, "question": "Generate more interview questions."}, return_only_outputs=True)
                        additional_questions = additional_questions_response["output_text"].split('\n')
                        additional_question_list = [question.strip() for question in additional_questions if question.strip()]

                        st.subheader("🎉 Additional Generated Interview Questions:")
                        for i, question in enumerate(additional_question_list):
                            st.write(f"{len(question_list) + i + 1}. {question}")

                        # Update session state with new questions
                        st.session_state.questions.extend(additional_question_list)
                        # Save additional questions to the file
                        save_questions_to_file(st.session_state.questions, questions_file_path)

    # Check if resume processing was successful before allowing video upload
    if 'questions' in st.session_state:
        st.subheader("📹 Upload Your Video Response")
        video_file = st.file_uploader("Upload a video answering the questions (MP4 format only)", type=["mp4"], accept_multiple_files=False)

        if video_file:
            # Check the file size
            max_size = 1 * 1024 * 1024 * 1024  # 1 GB
            if video_file.size > max_size:
                st.error("The video file size must be less than 1 GB.")
            else:
                # Save the uploaded video
                output_directory = 'resume_vid'  # Adjust the path as needed
                os.makedirs(output_directory, exist_ok=True)
                video_path = os.path.join(output_directory, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_file.name}")

                # Save the video file
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                st.success("Video uploaded successfully!")

                # Extract audio from the uploaded video without user notification
                audio_path = os.path.splitext(video_path)[0] + '.mp3'
                  # Call the function with the path of the extracted audio

                video_clip = VideoFileClip(video_path)
                video_clip.audio.write_audiofile(audio_path)
                video_clip.close()
                print(audio_path)
                process_audio(audio_path)  # Close the video clip


# Run the Streamlit app
if __name__ == "__main__":
    run_resume_interview_generator()

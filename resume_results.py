import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
import google.generativeai as genai
import pandas as pd  # Import pandas for data manipulation

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_file(file_path):
    """Read the content of a file and return it as a string."""
    with open(file_path, 'r') as file:
        return file.read()

def parse_answers(answers_text):
    """Parse the answers text to separate the transcript from scores."""
    # Use regex to split the transcript and the scores
    transcript_match = re.search(r"Transcript:\n(.*?)\n\n", answers_text, re.DOTALL)

    # Extract the specific scores using predefined patterns
    duration_match = re.search(r"Speech Duration:\s*([0-9.]+)\s*seconds", answers_text)
    wpm_match = re.search(r"Words Per Minute:\s*([0-9.]+)", answers_text)

    # Updated clarity_match to capture clarity score
    clarity_match = re.search(r"Clarity Score \(0 to 1\):\s*([0-1]\.\d+)", answers_text)

    snr_match = re.search(r"SNR \(dB\):\s*([0-9.]+)\s*dB", answers_text)

    # Initialize scores with empty strings
    scores = {
        "Speech Duration": duration_match.group(1) + " seconds" if duration_match else "",
        "Words Per Minute": wpm_match.group(1) if wpm_match else "",
        "Clarity Score": clarity_match.group(1) if clarity_match else "",
        "SNR": snr_match.group(1) + " dB" if snr_match else "",
    }

    # Check if transcript was found
    transcript = transcript_match.group(1).strip() if transcript_match else ""

    return transcript, scores

def get_evaluation_chain(questions, answers, duration, wpm, clarity, snr):
    prompt_template = """
    You're an evaluator, and these are the answers given by a user to the following questions. 
    Now, based on this, tell how the user can improve their way of talking or what things can be included in their answers.

    Questions: {questions}

    Answer Transcript: {answers}

    Evaluation Scores: 
    - Speech Duration: {duration}
    - Words Per Minute: {wpm}
    - Clarity Score: {clarity}
    - SNR: {snr}

    Your evaluation:
    Please provide specific feedback on how they can improve their speaking, such as whether to slow down their speech, raise or lower their voice, or work on clarity. 
    Also, give suggestions based on their scores, such as what to focus on for improvement, and provide links for resources to study or improve more on the questions and answers they were not able to give properly.
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["questions", "answers", "duration", "wpm", "clarity", "snr"])
    
    # Create the LLM chain directly
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def analyze_video():
    """Function to perform video analysis."""
    # Read questions and answers from the specified files
    questions = read_file(r"C:\Users\Hp\Downloads\BruteForce_TSEC\resume_processing\questions.txt")
    answers_text = read_file(r"C:\Users\Hp\Downloads\BruteForce_TSEC\resume_processing\answers.txt")

    # Parse the answers to get transcript and scores
    answers, scores = parse_answers(answers_text)

    # Extract scores
    duration = scores.get("Speech Duration", "")
    wpm = scores.get("Words Per Minute", "")
    clarity = scores.get("Clarity Score", "")
    snr = scores.get("SNR", "")

    # Create a DataFrame for displaying scores in a table
    scores_data = {
        "Metric": ["Speech Duration", "Words Per Minute", "Clarity Score", "SNR"],
        "Value": [duration, wpm, clarity, snr],
    }
    scores_df = pd.DataFrame(scores_data)

    # Get the evaluation chain
    evaluation_chain = get_evaluation_chain(questions, answers, duration, wpm, clarity, snr)

    # Run the evaluation chain and get the output with a spinner
    with st.spinner("Analyzing... Please wait."):
        output = evaluation_chain.run(
            questions=questions,
            answers=answers,
            duration=duration,
            wpm=wpm,
            clarity=clarity,
            snr=snr
        )

    # Display the scores and output feedback
    st.write("### Evaluation Scores")
    st.table(scores_df)
    st.write("### Evaluation Feedback:")
    st.write(output)

def main():
    st.title("Video Analysis Application")
    
    # Button to trigger video analysis
    if st.button("Analyze Video"):
        analyze_video()

# Execute the main function
if __name__ == "__main__":
    main()

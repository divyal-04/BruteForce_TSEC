import streamlit as st
import requests

# Function to fetch questions from the API
def fetch_questions(topic):
    url = f'https://aptitude-api.vercel.app/{topic}'
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()  # Expected to return a single dictionary
    else:
        st.error("Failed to fetch questions. Please try again.")
        return None

# Function for the Aptitude Question Generator
def aptitude_question_generator():
    # Available topics
    topics = [
        'Random',
        'MixtureAndAlligation',
        'ProfitAndLoss',
        'PipesAndCistern',
        'Age',
        'PermutationAndCombination',
        'SpeedTimeDistance',
        'Calendar',
        'SimpleInterest'
    ]

    st.header("Aptitude Question Generator")

    # Initialize session state for the question
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    # Dropdown for selecting the topic
    selected_topic = st.selectbox("Select a Topic", topics)

    # Fetch the question only when the button is clicked
    if st.button("Get Questions"):
        question = fetch_questions(selected_topic)
        st.session_state.current_question = question  # Store the question in session state

    # Check if there's a question in the session state to display
    if st.session_state.current_question:
        question = st.session_state.current_question

        # Display the question and options
        st.subheader(f"Question: {question['question']}")
        
        options = question['options']
        selected_option = st.radio("Choose your answer", options, key='answer_radio')
        
        # Display the correct answer and explanation when the Submit button is pressed
        if st.button("Submit Answer"):
            st.write(f"The correct answer is: {question['answer']}")
            st.write(f"Explanation: {question['explanation']}")
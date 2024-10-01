import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def teamwork_collaboration_scenario():
    # Context for generating the scenario
    context = f"""
    You have to generate the most in-depth valid working office scenarios on dealing with different personalities.
    Include simulated group projects where users must make decisions at key points, showing how their choices align with hypothetical teammates’ decisions.
    Incorporate AI-driven virtual teammates that simulate different types of collaborators (e.g., passive, dominant, cooperative), allowing users to practice adjusting their communication and collaboration styles.
    Offer role-based scenarios where users take on different responsibilities in a team, managing, delegating, or collaborating with virtual colleagues.
    Present tasks that require collaboration and ask users to make decisions as if they were part of a team, followed by a post-task analysis providing feedback on their collaboration style, leadership, or adaptability.
    Gamify the experience with quizzes or situational judgments to evaluate teamwork and collaboration skills, offering tailored feedback on areas for improvement.
    Create interactive problem-solving exercises where users can see how their approaches integrate with others’ approaches (e.g., real-time collaboration on code or project management tasks).
    Generate a real-life example. Don't provide a solution; just pose the question. Test the user's teamwork and collaboration skills. Pose only one scenario.
    """

    documents = [Document(page_content=context)]
    question = ""

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

    chain = get_conversational_chain()

    # Use Streamlit session state to store the scenario so it persists across user inputs
    if 'generated_scenario' not in st.session_state:
        st.session_state['generated_scenario'] = chain.run(input_documents=documents, question=question)

    # Display the generated scenario
    st.subheader("Teamwork and Collaboration Scenario")
    st.write(st.session_state['generated_scenario'])

    # User input for their response
    user_response = st.text_input("What would you do in this scenario?")

    if st.button("Submit Response"):
        if user_response:
            # Prepare the context for improving the user's response
            feedback_context = f"""
            Context:
            Based on the following scenario: {st.session_state['generated_scenario']}
            
            User's Response:
            {user_response}
            
            Suggest areas for improvement in the user's response.
            """

            feedback_documents = [Document(page_content=feedback_context)]
            
            # Define a new question for feedback
            feedback_question = "How can the user improve their response?"

            # Generate feedback
            feedback_chain = get_conversational_chain()
            improvement_suggestions = feedback_chain.run(input_documents=feedback_documents, question=feedback_question)

            # Display the improvement suggestions
            st.subheader("Suggestions for Improvement")
            st.write(improvement_suggestions)
        else:
            st.warning("Please provide a response before submitting.")

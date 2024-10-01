import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

def get_conversational_chain():
    """Create a conversational chain using the Google Generative AI model."""
    prompt_template = """
    Context:\n {context}\n
    Question:\n {question}\n

    Give relevant courses names and URLS from reputed sites like Coursera, Udemy or any more.
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(job_description):
    context = f"""
    Based on the following job description, suggest relevant online courses that can help prepare a candidate for this role:

    Job Description:
    {job_description}

    Please provide a list of recommended courses along with their providers.
    """

    documents = [Document(page_content=context)]
    question = ""

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": documents, "context": context, "question": question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def gemini_suggest_courses():
    st.header("Get Courses Acc To JD")

    user_question = st.text_input("Enter the Job Description")

    if user_question:
        user_input(user_question)

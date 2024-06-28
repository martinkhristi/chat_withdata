# Import necessary libraries
import streamlit as st
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
load_dotenv()

# Define functions to load language models
def load_groq_llm():
    return ChatGroq(model_name="Llama3-8b-8192", api_key=os.getenv('GROQ_API_KEY'))

def load_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4")

# Set up Streamlit page
st.set_page_config(page_title="Data Analysis Platform")

# Sidebar for user inputs
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
llm_choice = st.sidebar.selectbox("Select Language Model", ("Groq", "OpenAI"))

# Main application content starts here
st.title(" Data Analysis platform")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Display the first five rows of the data
    st.write("Here's a quick preview of the first five rows of your data:")
    st.write(data.head())

    # Load LLMs
    groq_llm = load_groq_llm()
    openai_llm = load_openai_llm()

    # SmartDataframe setup for language model interaction
    df_groq = SmartDataframe(data, config={'llm': groq_llm})
    df_openai = SmartDataframe(data, config={'llm': openai_llm})

    # User query input for natural language analysis
    query = st.text_input("Enter your query about the data:")
    if query:
        try:
            start_time = time.time()  # Record start time
            response = ""
            if llm_choice == "Groq":
                response = df_groq.chat(query)
            elif llm_choice == "OpenAI":
                response = df_openai.chat(query)
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            st.write(response)
            st.write(f"Time taken to answer the query: {elapsed_time:.2f} seconds")
        except Exception as e:
            st.error(f"An error occurred: {e}")

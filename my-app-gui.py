import streamlit as st
import pandas as pd
import io
import os
import codecs
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

def create_agent(api_key, csv_data, temperature=0, encoding="utf-8"):
    os.environ["OPENAI_API_KEY"] = api_key
    return create_csv_agent(OpenAI(temperature=temperature), csv_data, verbose=True, encoding=encoding)

def run_agent(agent, query):
    if agent is not None:
        return agent.run(query)
    else:
        return None

def main():
    st.title("CSV Assistant")
    
    st.markdown("""
    ## Instructions
    1. Enter your OpenAI API key in the sidebar.
    2. Upload a CSV file with the data you want the language model to use.
    3. Enter a query in the text box and the model will generate a response based on the CSV data.
    """)

    openai_api_key = st.sidebar.text_input(
        label="#### Your OpenAI API key 👇",
        placeholder="Paste your OpenAI API key, sk-",
        type="password")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    agent = None
    if openai_api_key and uploaded_file is not None:
        csv_data = io.StringIO()
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(codecs.iterdecode(uploaded_file, 'utf-8'), encoding="utf-8")
        else:
            st.write("File type not supported. Please upload a CSV file.")
        df.to_csv(csv_data, index=False)
        csv_data.seek(0)
        agent = create_agent(openai_api_key, csv_data)

    query = st.text_input("Enter your query")

    try:
        response = run_agent(agent, query)
        if response:
            st.write(response)
    except Exception as e:
        st.write(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

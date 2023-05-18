import streamlit as st
import io
import os
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

def create_agent(api_key, csv_data, temperature=0, encoding="utf-8"):
    os.environ["OPENAI_API_KEY"] = api_key
    return create_csv_agent(OpenAI(temperature=temperature), csv_data, verbose=True, encoding=encoding)

def run_agent(agent, query):
    return agent.run(query) if agent else None

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
    if openai_api_key and uploaded_file:
        csv_data = io.StringIO()
        try:
            # Read the uploaded CSV file and decode it as UTF-8
            content = uploaded_file.read().decode("utf-8")
            csv_data.write(content)
            csv_data.seek(0)
            
            # Create the language model agent using the OpenAI API key and the CSV data
            agent = create_agent(openai_api_key, csv_data)
        except UnicodeDecodeError:
            # If the file cannot be decoded as UTF-8, display an error message
            st.write("Error: Invalid encoding. Please upload a CSV file encoded in UTF-8.")
            return

    query = st.text_input("Enter your query")

    try:
        # Run the language model agent with the query
        response = run_agent(agent, query)
        if response:
            st.write(response)
    except Exception as e:
        st.write(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

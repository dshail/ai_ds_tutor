import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import nbformat
from nbconvert import PythonExporter
from nbclient import NotebookClient
from io import BytesIO
from fpdf import FPDF
from langchain_google import generativeai
from langchain.memory import FAISSRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load API key securely from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Setup FAISS Vector Memory
if os.path.exists("faiss_memory"):
    vectorstore = FAISS.load_local("faiss_memory")
else:
    vectorstore = FAISS.from_texts([""], [])

retriever = vectorstore.as_retriever()
memory = FAISSRetrieverMemory(retriever=retriever)

# Define AI Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Custom Prompt for Data Science Filtering
data_science_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    You are a Conversational AI Data Science Tutor. 
    You ONLY answer questions related to Data Science, Machine Learning, AI, Statistics, and related topics. 
    If a question is not related to Data Science, politely refuse to answer.
    
    Question: {input}
    Answer:
    """
)

conversation = ConversationChain(llm=llm, memory=memory, prompt=data_science_prompt)

# Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("🤖📊 AI Data Science Tutor")
st.write("Ask me anything about Data Science!")

# Sidebar Features
st.sidebar.title("📂 Features")
features = ["Live Python Code Execution", "Data Visualization", "Real-time API Data", "AI Study Notes", "Jupyter Notebook Integration", "AI Code Debugging", "Custom Model Training"]
selected_feature = st.sidebar.radio("Choose a Feature:", features)

# Function: Execute Python Code
def execute_python_code(code):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals
    except Exception as e:
        return str(e)

# Function: AI Debugging Assistance
def debug_python_code(code):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "No errors detected. Code executed successfully."
    except Exception as e:
        return f"Error detected: {str(e)}. Suggested fix: {llm.predict(input=f'Debug the following code: {code}') }"

# Function: Train a Custom Model
def train_custom_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# Function: Integrate Jupyter Notebook
def execute_notebook(nb_code):
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(nb_code))
    client = NotebookClient(nb, execute=True)
    client.execute()
    exporter = PythonExporter()
    code, _ = exporter.from_notebook_node(nb)
    return code

# Handle Feature Selection
if selected_feature == "Live Python Code Execution":
    user_code = st.text_area("Write Python Code:")
    if st.button("Run Code"):
        output = execute_python_code(user_code)
        st.write(output)

elif selected_feature == "Data Visualization":
    st.write("Generated Data Visualization:")
    df = pd.DataFrame({"Category": ["A", "B", "C", "D"], "Values": [23, 45, 56, 78]})
    fig, ax = plt.subplots()
    sns.barplot(x="Category", y="Values", data=df, ax=ax)
    st.pyplot(fig)

elif selected_feature == "Real-time API Data":
    if st.button("Fetch API Data"):
        data = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json").json()
        st.json(data)

elif selected_feature == "AI Study Notes":
    st.markdown("""
    **Data Science Quick Study Notes**
    - **Machine Learning**: Supervised, Unsupervised, Reinforcement Learning.
    - **Deep Learning**: Neural Networks, CNNs, RNNs.
    - **Statistics**: Mean, Median, Variance, Hypothesis Testing.
    """)

elif selected_feature == "Jupyter Notebook Integration":
    nb_code = st.text_area("Write Jupyter Notebook Code:")
    if st.button("Execute Notebook"):
        output = execute_notebook(nb_code)
        st.code(output)

elif selected_feature == "AI Code Debugging":
    debug_code = st.text_area("Write Python Code for Debugging:")
    if st.button("Debug Code"):
        debug_output = debug_python_code(debug_code)
        st.write(debug_output)

elif selected_feature == "Custom Model Training":
    uploaded_file = st.file_uploader("Upload CSV for Model Training:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        accuracy = train_custom_model(df)
        st.write(f"Model Accuracy: {accuracy:.2f}")

if user_input := st.chat_input("Type your question here..."):
    response = conversation.predict(input=user_input)
    st.write(response)
    vectorstore.add_texts([user_input, response])
    vectorstore.save_local("faiss_memory")

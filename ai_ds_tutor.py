import streamlit as st
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load API Key securely from Streamlit Cloud Secrets
api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Memory for conversation history
memory = ConversationBufferMemory()

# Conversation Chain
conversation = ConversationChain(llm=llm, memory=memory)

# Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", page_icon="📊")
st.title("📊 AI Conversational Data Science Tutor")
st.markdown("Ask me anything related to Data Science!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_input = st.chat_input("Ask a data science question...")

if user_input:
    # Store user query
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get AI response
    response = conversation.run(user_input)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display response
    st.chat_message("assistant").write(response)

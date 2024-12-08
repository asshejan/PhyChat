import streamlit as st
import os
import asyncio
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Ensure necessary directories
os.makedirs('pdfFiles', exist_ok=True)
os.makedirs('vectorDB', exist_ok=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'template' not in st.session_state:
    # Strictly constraining responses to PDF content
    st.session_state.template = """You are a physics assistant that answers questions strictly based on the content provided from a textbook PDF.
If the information requested is not found in the textbook, respond with: "I'm sorry, I can only provide answers based on the textbook."

Context (from textbook only): {context}
User History: {history}

User: {question}
Assistant (based on textbook content only):"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

# Define the path to the PDF file
fixed_pdf_path = 'phybook10.pdf'

# Initialize vectorstore only once with cached documents
if 'vectorstore' not in st.session_state:
    if os.path.exists('vectorDB'):
        st.session_state.vectorstore = Chroma(
            persist_directory='vectorDB',
            embedding_function=OllamaEmbeddings(model="llama3.1")
        )
    else:
        # Load and split the PDF book
        loader = PyPDFLoader(fixed_pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for faster retrieval
            chunk_overlap=50,  # Minimal overlap
            length_function=len
        )

        all_splits = text_splitter.split_documents(data)

        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="llama3.1")
        )

        st.session_state.vectorstore.persist()

# Set up the retriever with minimal overlap for efficient retrieval
st.session_state.retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

# Initialize the LLM model with asynchronous response
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3.1",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.session_state.prompt,
            "memory": st.session_state.memory,
        }
    )

# Set up Streamlit interface
st.title("PhyChat: A Physics Book-Based Chatbot")

# Show only the last 5 messages for a cleaner chat interface
for message in st.session_state.chat_history[-5:]:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Async function to handle user input and chatbot response
async def get_response(user_input):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("PhyChat is thinking..."):
            # Get response asynchronously
            response = st.session_state.qa_chain(user_input)
        
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            await asyncio.sleep(0.01)  # Shorter delay for faster typing effect
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)

# Process user input
if user_input := st.chat_input("Ask a question about the textbook:", key="user_input"):
    asyncio.run(get_response(user_input))  # Run async response handling

import streamlit as st
import os
import pandas as pd
import time
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
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Ensure required directories exist
os.makedirs('pdfFiles', exist_ok=True)
os.makedirs('vectorDB', exist_ok=True)

# Load CSV with questions and answers for evaluation
qa_df = pd.read_csv('physics_short.csv')  # Update the path to your CSV file
questions = qa_df['Question'].tolist()
answers = qa_df['Answer'].tolist()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'template' not in st.session_state:
    st.session_state.template = """You are a helpful assistant answering questions only based on the content from a provided physics textbook. 
If the answer is not directly found in the textbook context, respond with "I'm sorry, I can only answer based on the provided textbook content."

    Context (from textbook only): {context}
    User History: {history}

    User: {question}
    Chatbot (based on textbook content only):"""

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

# Define PDF file path
fixed_pdf_path = 'phybook10.pdf'

# Initialize vectorstore only once
if 'vectorstore' not in st.session_state:
    if os.path.exists('vectorDB'):
        st.session_state.vectorstore = Chroma(
            persist_directory='vectorDB',
            embedding_function=OllamaEmbeddings(model="llama3.1")
        )
    else:
        loader = PyPDFLoader(fixed_pdf_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len
        )

        all_splits = text_splitter.split_documents(data)

        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="llama3.1")
        )

        st.session_state.vectorstore.persist()

st.session_state.retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5},
)

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

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to evaluate response using ROUGE
def evaluate_rouge(reference_answer, chatbot_answer):
    scores = scorer.score(reference_answer, chatbot_answer)
    return {
        "ROUGE-1 F1": scores['rouge1'].fmeasure,
        "ROUGE-2 F1": scores['rouge2'].fmeasure,
        "ROUGE-L F1": scores['rougeL'].fmeasure
    }

# Function to evaluate response using BLEU
def evaluate_bleu(reference_answer, chatbot_answer):
    reference_tokens = reference_answer.split()
    response_tokens = chatbot_answer.split()
    score = sentence_bleu([reference_tokens], response_tokens)
    return {"BLEU": score}

# Array to store all ROUGE and BLEU results
results = []

# Process all questions and calculate ROUGE and BLEU scores
for question, reference_answer in zip(questions, answers):
    response = st.session_state.qa_chain(question)
    chatbot_answer = response['result']
    
    # Evaluate response with ROUGE
    rouge_scores = evaluate_rouge(reference_answer, chatbot_answer)
    # Evaluate response with BLEU
    bleu_score = evaluate_bleu(reference_answer, chatbot_answer)
    
    results.append({
        "Question": question,
        "Reference Answer": reference_answer,
        "Chatbot Answer": chatbot_answer,
        "ROUGE-1 F1": rouge_scores["ROUGE-1 F1"],
        "ROUGE-2 F1": rouge_scores["ROUGE-2 F1"],
        "ROUGE-L F1": rouge_scores["ROUGE-L F1"],
        "BLEU": bleu_score["BLEU"]
    })

# Display all results as a DataFrame
results_df = pd.DataFrame(results)
st.write("ROUGE and BLEU Scores for All Questions:")
st.write(results_df)

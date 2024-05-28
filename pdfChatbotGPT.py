#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# IMPORT LIBRARIES
from personal import secrets
import os
import streamlit as st
import pandas as pd
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# OPENAI KEY 
api_key = secrets.get('OPENAI API KEY')
os.environ["OPENAI_API_KEY"] = str(api_key)


# WEB PAGE SETUP

# Initialize session state to store conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if "messages" not in st.session_state:
    st.session_state.messages = [] 
    
# set up main page
st.title("GLANSIS PDF Chatbot")
st.write("This chatbot interface will allow GLANSIS users to ask PDFs a series of questions.")


# set up sidebar
st.sidebar.title("Get Invasive Species Environmental Impacts")

pdf = st.sidebar.file_uploader("Select PDF")

species_name = st.sidebar.text_input("Enter Species Names")
st.sidebar.button('Run')


# code to process PDF
if pdf is not None:
    try: 
        # Read in text from PDF 
        raw_text = ''
        pdf_doc = PdfReader(pdf)
        for page in pdf_doc.pages:
            raw_text += page.extract_text()
            
    except:
        st.error("Please add PDF")
        
    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    
    text_chunk = text_splitter.split_text(raw_text)
    
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    # Create vector database
    docsearch = FAISS.from_texts(text_chunk, embedding=embeddings)
    st.session_state["docsearch"] = docsearch
    
    # Question requirements
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type = 'stuff')
        
    # Display success message in chat interface
    st.success("PDF successfully processed! You can start asking questions.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("What are your questions?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)  
        
    docsearch = st.session_state.get("docsearch")
    
    if docsearch:
        docs = docsearch.similarity_search(prompt)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type = 'stuff')
        response = chain.run(input_documents=docs, question=prompt)
    else:
        response = "Please upload a PDF first."
        
    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})


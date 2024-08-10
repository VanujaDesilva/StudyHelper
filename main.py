# import necessary libraries
import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()  # take enviroment varialbel

# UI
st.title("📚 StudyHelper 🤖")
st.subheader("Your Intelligent Study Companion")
st.sidebar.title("Study Article URLs")

article_urls = []
for i in range(3):  # because we are getting only 3 URLs at a time
    url = st.sidebar.text_input(f"Article {i + 1} URL")
    article_urls.append(url)

process_url_click = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"

if process_url_click:
    # loading articles' data
    article_loader = UnstructuredURLLoader(urls=article_urls)
    data = article_loader.load()
    # splitting data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    # gathering all the data chunks
    docs = text_splitter.split_documents(data)
    # create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    # Save the FAISS index to a pickle file
    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai, f)


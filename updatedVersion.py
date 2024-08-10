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

load_dotenv()  # Load environment variables

# UI Setup with Enhanced Design
st.set_page_config(page_title="StudyHelper", page_icon="📚", layout="centered")

# Custom CSS for enhanced design
st.markdown("""
    <style>
    body {
        background-image: url('https://www.pixelstalk.net/wp-content/uploads/images6/HD-Study-Desktop-Wallpaper.jpg');
        background-size: cover;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5 {
        color: #fff;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 StudyHelper 🤖")
st.subheader("Your Intelligent Study Companion")

st.sidebar.title("Study Article URLs")

article_urls = []
for i in range(3):  # Gather up to 3 URLs
    url = st.sidebar.text_input(f"Article {i + 1} URL")
    article_urls.append(url)

process_url_click = st.sidebar.button("Process URLs")

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_click:
    try:
        # Load and process the articles
        main_placeholder.text("Loading and processing the articles...")

        article_loader = UnstructuredURLLoader(urls=article_urls)
        data = article_loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000  # Reduced chunk size for fewer tokens
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding process completed.")

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

        main_placeholder.text("Process completed successfully.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        time.sleep(2)  # Adding delay to prevent rate limit issues

# Query section with results display
query = main_placeholder.text_input("Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorestore = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorestore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)

# Revision Section for MCQs
if process_url_click:
    generate_revision = st.button("Generate Revision MCQs")
    if generate_revision:
        try:
            # Generate MCQs using OpenAI
            mcq_prompt = "Generate 10 multiple-choice questions based on the following content:\n"
            for doc in docs:
                mcq_prompt += doc.page_content + "\n"

            mcq_response = llm(mcq_prompt)["choices"][0]["text"]
            mcqs = mcq_response.strip().split("\n\n")

            score = 0
            for i, mcq in enumerate(mcqs):
                question, *choices = mcq.split("\n")
                user_answer = st.radio(question, choices, key=i)
                if "Answer:" in mcq:
                    correct_answer = mcq.split("Answer: ")[-1].strip()
                    if user_answer == correct_answer:
                        score += 1

            st.write(f"Your score: {score}/{len(mcqs)}")
        except Exception as e:
            st.error(f"An error occurred while generating MCQs: {e}")
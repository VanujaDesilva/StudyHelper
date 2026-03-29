"""
StudyHelper - AI-Powered Article Q&A and Revision Chatbot
=========================================================
A RAG (Retrieval-Augmented Generation) application that allows users to
provide article URLs, ask questions grounded in article content, and
test their understanding through auto-generated multiple-choice quizzes.

Tech stack: Streamlit, LangChain, OpenAI, FAISS
Author:     <your-name>
"""

import os
import re
import json
import logging
from urllib.parse import urlparse

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API keys and other secrets from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All tuneable parameters are gathered here so they are easy to adjust
# without digging through business logic.

FAISS_INDEX_DIR = "faiss_index"       # Directory where the FAISS index is persisted
MAX_URLS        = 3                   # Maximum number of article URLs accepted
CHUNK_SIZE      = 1000                # Characters per document chunk
CHUNK_OVERLAP   = 200                 # Overlap between consecutive chunks
QA_MODEL        = "gpt-3.5-turbo"    # Model used for question answering
MCQ_MODEL       = "gpt-3.5-turbo"    # Model used for MCQ generation
QA_TEMPERATURE  = 0.2                 # Low temperature for factual Q&A answers
MCQ_TEMPERATURE = 0.4                 # Slightly higher for creative question phrasing
MCQ_COUNT       = 5                   # Number of MCQs to generate per session

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def is_valid_url(url: str) -> bool:
    """
    Check whether a string is a well-formed HTTP or HTTPS URL.
    Returns False for empty strings, malformed input, or non-HTTP schemes.
    """
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def init_session_state() -> None:
    """
    Initialise Streamlit session-state keys with sensible defaults.

    Streamlit reruns the entire script on every user interaction, so all
    data that must survive across reruns is stored in st.session_state.
    """
    defaults = {
        "articles_processed": False,   # Whether articles have been indexed
        "docs": [],                    # Document chunks from the last ingestion
        "mcqs": [],                    # Generated MCQ dicts
        "user_answers": {},            # Map of question index -> selected option
        "quiz_submitted": False,       # Whether the user has submitted the quiz
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Core RAG Pipeline
# ---------------------------------------------------------------------------

def load_and_split_articles(urls: list[str]) -> list:
    """
    Fetch article content from the provided URLs and split it into
    overlapping text chunks suitable for embedding.

    Steps:
        1. UnstructuredURLLoader fetches and parses each URL.
        2. RecursiveCharacterTextSplitter breaks the documents into chunks
           of CHUNK_SIZE characters with CHUNK_OVERLAP overlap so that
           context is not lost at chunk boundaries.

    Raises ValueError if no content could be extracted.
    """
    loader = UnstructuredURLLoader(urls=urls)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError("No content could be extracted from the provided URLs.")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(raw_docs)


def build_vectorstore(docs: list) -> FAISS:
    """
    Create a FAISS vector store from document chunks and save it to disk.

    Each chunk is embedded using OpenAI's text-embedding model. The
    resulting vectors are indexed with FAISS for fast similarity search.
    The index is persisted via FAISS's native save_local() method (not
    pickle) for safety and cross-version compatibility.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def ask_question(query: str) -> dict:
    """
    Answer a user question using a RetrievalQA chain.

    Flow:
        1. Load the FAISS index.
        2. Retrieve the top-k most relevant chunks for the query.
        3. Pass the chunks + question to the LLM via LangChain's
           RetrievalQAWithSourcesChain.
        4. Return a dict with 'answer' and 'sources' keys.
    """
    llm = ChatOpenAI(model=QA_MODEL, temperature=QA_TEMPERATURE, max_tokens=1024)
    vectorstore = load_vectorstore()
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )
    return chain.invoke({"question": query})


# ---------------------------------------------------------------------------
# MCQ Generation
# ---------------------------------------------------------------------------

# System prompt that instructs the LLM to return structured JSON.
# Using a strict schema in the prompt ensures we can parse the output
# programmatically rather than relying on fragile text splitting.
MCQ_SYSTEM_PROMPT = f"""You are an expert quiz generator for students.
Given the article content below, generate exactly {MCQ_COUNT} multiple-choice questions.

RULES:
- Each question must have exactly 4 options labelled A, B, C, D.
- Exactly one option must be correct.
- Questions should test understanding, not trivial details.
- Return ONLY valid JSON. No markdown, no code fences, no extra text.

Return a JSON array where each element has this schema:
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "answer": "A"
}}
"""


def generate_mcqs(docs: list) -> list[dict]:
    """
    Generate structured MCQs from article content using the LLM.

    Steps:
        1. Concatenate the first 10 document chunks (trimmed to ~6000 chars)
           to stay within the model's context window.
        2. Send a system prompt + article content to the LLM.
        3. Strip any accidental markdown fences from the response.
        4. Parse the JSON and validate each question has the required fields.

    Returns a list of dicts, each with keys: question, options, answer.
    """
    # Concatenate chunk text, capped to avoid exceeding token limits
    content = "\n".join(doc.page_content for doc in docs[:10])
    content = content[:6000]

    llm = ChatOpenAI(model=MCQ_MODEL, temperature=MCQ_TEMPERATURE, max_tokens=2048)

    messages = [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": f"Article content:\n\n{content}"},
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Some models wrap JSON output in ```json ... ``` fences; strip them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    mcqs = json.loads(raw)

    # Validate structure: keep only well-formed questions
    validated: list[dict] = []
    for q in mcqs:
        has_required_keys = (
            isinstance(q, dict)
            and "question" in q
            and "options" in q
            and "answer" in q
        )
        has_four_options = has_required_keys and len(q["options"]) == 4
        if has_four_options:
            validated.append(q)

    return validated


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[list[str], bool]:
    """
    Render the sidebar with URL input fields and the process button.
    Returns a tuple of (list of non-empty URLs, whether the button was clicked).
    """
    st.sidebar.title("Article URLs")
    st.sidebar.caption("Paste up to 3 article URLs to study from.")

    urls: list[str] = []
    for i in range(MAX_URLS):
        url = st.sidebar.text_input(
            f"URL {i + 1}",
            key=f"url_{i}",
            placeholder="https://example.com/article",
        )
        # Only collect non-empty inputs
        if url and url.strip():
            urls.append(url.strip())

    clicked = st.sidebar.button("Process Articles", use_container_width=True)

    # Persistent status indicator so the user knows articles are loaded
    if st.session_state.articles_processed:
        st.sidebar.success("Articles loaded and indexed.")

    return urls, clicked


def render_qa_section() -> None:
    """
    Render the Q&A tab where users type a question and receive an
    answer grounded in the indexed article content.
    """
    st.subheader("Ask a Question")
    query = st.text_input(
        "Type your question about the articles:",
        placeholder="e.g. What are the key findings discussed in the article?",
    )

    if not query:
        return

    with st.spinner("Retrieving answer..."):
        try:
            result = ask_question(query)
        except Exception as exc:
            st.error(f"Failed to get an answer: {exc}")
            return

    # Display the answer
    st.markdown("**Answer**")
    st.write(result.get("answer", "No answer returned."))

    # Display source URLs if the chain returned any
    sources = result.get("sources", "")
    if sources and sources.strip():
        with st.expander("Sources"):
            for src in sources.strip().split("\n"):
                if src.strip():
                    st.write(f"- {src.strip()}")


def render_revision_section() -> None:
    """
    Render the Revision tab where MCQs are generated, presented as a
    form, scored on submission, and feedback is shown per question.
    """
    st.subheader("Revision Mode")

    # Action buttons: Generate / Reset
    col_generate, col_reset = st.columns(2)
    with col_generate:
        generate_btn = st.button("Generate MCQs", use_container_width=True)
    with col_reset:
        if st.session_state.mcqs:
            if st.button("Reset Quiz", use_container_width=True):
                st.session_state.mcqs = []
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

    # Handle MCQ generation
    if generate_btn:
        with st.spinner("Generating questions from your articles..."):
            try:
                mcqs = generate_mcqs(st.session_state.docs)
                if not mcqs:
                    st.warning("Could not generate valid MCQs. Try different articles.")
                    return
                st.session_state.mcqs = mcqs
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
            except json.JSONDecodeError:
                st.error("The AI returned an unexpected format. Please try again.")
                return
            except Exception as exc:
                st.error(f"Error generating MCQs: {exc}")
                return

    # Nothing to show if no MCQs have been generated yet
    if not st.session_state.mcqs:
        return

    # Render the quiz inside a Streamlit form so all answers are
    # collected before submission (avoids a rerun per radio click).
    with st.form("mcq_form"):
        for idx, mcq in enumerate(st.session_state.mcqs):
            st.markdown(f"**Q{idx + 1}. {mcq['question']}**")
            options = [f"{key}: {value}" for key, value in mcq["options"].items()]
            choice = st.radio(
                "Select your answer:",
                options,
                key=f"mcq_{idx}",
                label_visibility="collapsed",
            )
            st.session_state.user_answers[idx] = choice

        submitted = st.form_submit_button("Submit Answers", use_container_width=True)

    if submitted:
        st.session_state.quiz_submitted = True

    # Score and display results after submission
    if not st.session_state.quiz_submitted:
        return

    score = 0
    st.markdown("**Results**")

    for idx, mcq in enumerate(st.session_state.mcqs):
        correct_key = mcq["answer"]
        user_choice = st.session_state.user_answers.get(idx, "")
        # Extract the letter before the colon (e.g. "A" from "A: some text")
        user_key = user_choice.split(":")[0] if user_choice else ""

        if user_key == correct_key:
            score += 1
            st.success(f"Q{idx + 1}: Correct")
        else:
            correct_text = mcq["options"].get(correct_key, "")
            st.error(f"Q{idx + 1}: Incorrect — Answer: {correct_key}: {correct_text}")

    # Summary
    total = len(st.session_state.mcqs)
    percentage = int((score / total) * 100) if total else 0
    st.markdown(f"### Score: {score} / {total} ({percentage}%)")

    if percentage == 100:
        st.balloons()
    elif percentage >= 60:
        st.info("Good job! Review the ones you missed and try again.")
    else:
        st.warning("Keep studying. Re-read the articles and retake the quiz.")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Top-level function that wires together the sidebar, article
    processing pipeline, and the two main tabs (Q&A and Revision).
    """
    st.set_page_config(page_title="StudyHelper", layout="centered")

    st.title("StudyHelper")
    st.caption(
        "AI-powered study companion — ask questions about any article "
        "and test yourself with auto-generated quizzes."
    )

    # Ensure all session-state keys exist before any UI renders
    init_session_state()

    # ---- Sidebar: URL input and processing trigger ----
    urls, process_clicked = render_sidebar()

    # ---- Article processing pipeline ----
    if process_clicked:
        # Separate valid from invalid URLs and warn accordingly
        valid_urls = [u for u in urls if is_valid_url(u)]
        invalid_urls = [u for u in urls if u and not is_valid_url(u)]

        if not valid_urls:
            st.error("Please provide at least one valid URL (https://...).")
            return

        if invalid_urls:
            st.warning(f"Skipped invalid URLs: {', '.join(invalid_urls)}")

        # Show a collapsible status panel while processing
        progress = st.status("Processing articles...", expanded=True)

        try:
            progress.write("Loading articles...")
            docs = load_and_split_articles(valid_urls)

            progress.write(f"Split into {len(docs)} chunks.")
            progress.write("Building vector index...")
            build_vectorstore(docs)

            # Persist results in session state
            st.session_state.articles_processed = True
            st.session_state.docs = docs

            # Reset any previous quiz when new articles are loaded
            st.session_state.mcqs = []
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False

            progress.update(label="Articles processed successfully.", state="complete")

        except Exception as exc:
            logger.exception("Error processing articles")
            progress.update(label="Processing failed.", state="error")
            st.error(f"Something went wrong: {exc}")
            return

    # ---- Main content area ----
    if st.session_state.articles_processed:
        # Two tabs: one for free-form Q&A, one for structured revision
        tab_qa, tab_revision = st.tabs(["Q&A", "Revision"])
        with tab_qa:
            render_qa_section()
        with tab_revision:
            render_revision_section()
    else:
        st.info(
            "Paste article URLs in the sidebar and click "
            "Process Articles to get started."
        )


if __name__ == "__main__":
    main()

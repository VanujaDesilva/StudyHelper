# StudyHelper — AI-Powered Article Q&A and Revision Chatbot

An intelligent study companion that lets you paste article URLs, ask questions about their content using RAG (Retrieval-Augmented Generation), and test your understanding with auto-generated MCQs.

## Features

- **Article Ingestion** — Paste up to 3 article URLs; the app extracts, chunks, and indexes the content automatically.
- **Q&A with Sources** — Ask any question and get grounded answers with source attribution, powered by a retrieval chain over a FAISS vector store.
- **Revision Mode** — Auto-generates multiple-choice questions from the article content. Answer them interactively, get scored, and see corrections for wrong answers.

## Architecture

```
User --> Streamlit UI
              |
              |-- URL Ingestion --> UnstructuredURLLoader --> RecursiveCharacterTextSplitter
              |                                                       |
              |                                               OpenAI Embeddings
              |                                                       |
              |                                                 FAISS Vector Store
              |                                                       |
              |-- Q&A Tab ---------> RetrievalQAWithSourcesChain -----+
              |                          (ChatOpenAI + FAISS Retriever)
              |
              +-- Revision Tab ----> ChatOpenAI (structured JSON prompt) --> Interactive Quiz
```

### Key Design Decisions

| Concern | Approach |
|---|---|
| Vector storage | FAISS with native `save_local` / `load_local` (no pickle) |
| LLM | `ChatOpenAI` (chat-completions API) instead of legacy completions |
| MCQ generation | Structured JSON schema in system prompt with validation and fallback |
| State management | `st.session_state` to persist data across Streamlit reruns |
| Error handling | Try/except around every I/O and LLM call with user-friendly messages |

## Tech Stack

- **Frontend:** Streamlit
- **LLM Orchestration:** LangChain
- **Embeddings and LLM:** OpenAI API (`text-embedding-ada-002`, `gpt-3.5-turbo`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Document Loading:** LangChain `UnstructuredURLLoader`

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/studyhelper.git
cd studyhelper

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and paste your OpenAI key
```

### Run

```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`.

### Usage

1. Paste 1-3 article URLs in the sidebar.
2. Click **Process Articles** and wait for indexing to complete.
3. Switch to the **Q&A** tab to ask questions.
4. Switch to the **Revision** tab, click **Generate MCQs**, answer them, and submit.

## Project Structure

```
studyhelper/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

## Possible Enhancements

- Swap FAISS for a persistent vector DB (Pinecone, Weaviate, Chroma) so articles survive across sessions.
- Add PDF and file upload support alongside URLs.
- Let users choose difficulty level and question count for MCQs.
- Add conversation memory for multi-turn Q&A.
- Deploy on Streamlit Community Cloud or AWS.

## License

MIT

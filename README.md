# AI Resume Analyzer with RAG + Semantic Search Chatbot

This project is a complete end-to-end solution for HR teams to analyze resumes using semantic search and a Retrieval-Augmented Generation (RAG) chatbot.

## 🚀 Features
* **Resume Upload:** Supports PDF and DOCX files.
* **Semantic Search:** Find the best candidates for a job description using FAISS and sentence embeddings.
* **Hybrid Search:** Combines vector similarity with keyword boosting.
* **RAG Chatbot:** Ask questions about your candidates and get answers based on their resume content.
* **Metadata Handling:** Tracks filenames and chunk IDs for traceability.
* **Streamlit UI:** Easy-to-use 3-page interface.

## 🛠️ Tech Stack
* **Python**
* **LangChain:** For document loading, splitting, and RAG logic.
* **FAISS:** Vector database for efficient similarity search.
* **Sentence Transformers:** `all-MiniLM-L6-v2` for high-quality embeddings.
* **Streamlit:** Frontend UI.
* **Groq / LLaMA 3:** For the RAG-based chatbot responses.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd resume-analyzer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys:**
   * Copy `.env.example` to `.env`.
   * Get a free API key from the Groq Console.
   * Add your key to the `.env` file:
     ```env
     GROQ_API_KEY=your_key_here
     ```

## 💻 How to Run

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Using the app:**
   * **Page 1 (Upload):** Upload at least 10 resumes (PDF/DOCX) and click "Process & Index".
   * **Page 2 (Job Matching):** Paste a job description to find the most relevant candidates.
   * **Page 3 (Chat):** Ask specific questions like *"Who has experience with Kubernetes?"* or *"Summarize John Doe's projects"*.

## 📂 Folder Structure
* `data/` – Stores uploaded resumes locally.
* `vectorstore/` – Stores the FAISS index and metadata.
* `app.py` – Main Streamlit application.
* `loader.py` – Document loading and text splitting.
* `embeddings.py` – Embedding generation and FAISS management.
* `retriever.py` – Similarity and hybrid search logic.
* `chatbot.py` – LLM integration and RAG chain.
* `utils.py` – Helper functions.
* `requirements.txt` – List of dependencies.

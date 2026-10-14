# RAG Document Assistant

A document question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions based on the content of uploaded PDF documents.

## Overview

Large language models may generate incorrect answers or provide information that is not present in a particular document. This project uses document retrieval to provide relevant text as context before generating an answer.

Users upload a PDF, ask a question, and receive an answer based on the retrieved document sections.

## Features

- **PDF Upload:** Accepts PDF documents through an interactive interface.
- **Text Extraction:** Extracts text from PDF pages using PyPDF.
- **Text Chunking:** Splits extracted text into smaller sections.
- **Embeddings:** Uses Sentence Transformers to represent text as numerical vectors.
- **Vector Search:** Uses FAISS to retrieve relevant document chunks.
- **AI-Generated Answers:** Uses the Gemini API to generate answers from retrieved context.
- **Source Context Display:** Allows users to inspect the retrieved document sections.
- **Out-of-Context Handling:** Instructs the model to report when information is unavailable in the provided context.

## Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Google Gemini API
- Google Gen AI Python SDK
- python-dotenv

## How It Works

1. The user uploads a PDF document.
2. PyPDF extracts the text from its pages.
3. The application divides the text into chunks.
4. Sentence Transformers converts the chunks into embeddings using `all-MiniLM-L6-v2`.
5. FAISS indexes the embeddings for similarity-based retrieval.
6. The user's question is converted into an embedding.
7. FAISS retrieves the most relevant chunks.
8. The retrieved chunks and question are sent to Gemini as context.
9. Gemini generates an answer, which is displayed in the Streamlit interface.

## RAG Pipeline

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Index
    ↓
Question Embedding
    ↓
Relevant Chunk Retrieval
    ↓
Gemini Answer Generation
    ↓
Answer + Retrieved Context
```

## Technologies Explained

- **Retrieval-Augmented Generation (RAG):** Combines information retrieval with language generation.
- **Embeddings:** Numerical representations of text used to compare semantic similarity.
- **FAISS:** A library for efficient similarity search over vectors.
- **Sentence Transformers:** Generates text embeddings using a pretrained model.
- **Gemini:** Generates a natural-language answer using the retrieved context.

## Project Structure

```text
RAG-Document-Assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

The `.env` file stores the local API key and is excluded from version control.

## Setup and Installation

### Prerequisites

- Python installed
- A Google Gemini API key
- Internet access for API requests and initial model download

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the example values with your repository URL and folder name.

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a local `.env` file in the project root:

```text
GEMINI_API_KEY=your_actual_api_key
```

Never commit the actual `.env` file or API key.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, upload a PDF, and ask a question about its contents.

## Example Use Case

A user uploads an electronics laboratory experiment PDF about ASK modulation and demodulation.

**Example question:** What happens when the DATA input is changed using the DIP switch?

The application retrieves relevant document text and uses it to generate an answer based on that context.

If a question asks for information absent from the retrieved context, the prompt instructs the model to say that the information is unavailable in the document.

## Limitations

- Answer quality depends on the PDF's extractable text and document formatting.
- The current implementation uses fixed-size character chunks rather than semantic chunking.
- FAISS uses `IndexFlatL2` for vector search.
- Only a limited number of chunks are retrieved for each question.
- Retrieval can miss relevant information, and generated answers are not guaranteed to be correct.
- Scanned PDFs may require OCR, which is not implemented in this version.
- The current application processes the uploaded document during the session rather than maintaining a persistent document index.

## Future Improvements

- Add OCR support for scanned PDFs.
- Improve chunking with overlap and semantic boundaries.
- Display page numbers alongside retrieved passages.
- Support multiple documents and persistent indexes.
- Add retrieval evaluation and answer-quality checks.
- Add downloadable answers with source references.

## Learning Outcomes

This project provides practical experience with document processing, text embeddings, vector similarity search, Retrieval-Augmented Generation, API integration, and interactive application development.

## Disclaimer

This is an educational prototype. Answers should be checked against the original document, especially when accuracy is important.
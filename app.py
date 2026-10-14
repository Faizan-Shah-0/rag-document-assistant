import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from dotenv import load_dotenv
import os


# -----------------------------
# Load API key
# ----------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# -----------------------------
# Page configuration
# ----------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Assistant")

st.write(
    "Upload a PDF and ask questions based on its content."
)


# -----------------------------
# Upload PDF
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # -----------------------------
    # Extract text
    # ----------------------------

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    st.success("PDF uploaded successfully!")

    st.write("Number of pages:", len(reader.pages))


    # -----------------------------
    # Create chunks
    # ----------------------------

    chunk_size = 500

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)


    st.write("Number of chunks:", len(chunks))


    # -----------------------------
    # Create embeddings
    # ----------------------------

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)


    # -----------------------------
    # Create FAISS index
    # ----------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)


    st.success("Document processed successfully!")


    # -----------------------------
    # Ask question
    # ----------------------------

    question = st.text_input(
        "Ask a question about the document:"
    )


    if question:

        # -----------------------------
        # Convert question to embedding
        # -----------------------------

        question_embedding = embedding_model.encode(
            [question]
        )


        # -----------------------------
        # Search for relevant chunks
        # -----------------------------

        distances, indices = index.search(
            question_embedding,
            k=min(3, len(chunks))
        )


        relevant_chunks = []

        for i in indices[0]:

            relevant_chunks.append(
                chunks[i]
            )


        # -----------------------------
        # Create context
        # -----------------------------

        context = "\n\n".join(
            relevant_chunks
        )


        # -----------------------------
        # Send context + question to Gemini
        # -----------------------------

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer cannot be found in the context,
say that the information is not available in the document.

Document Context:
{context}

User Question:
{question}

Give a clear and concise answer.
"""


        with st.spinner("🤖 Generating answer..."):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                answer = response.text

                st.subheader("💬 Answer")

                st.write(answer)


                # -----------------------------
                # Show retrieved chunks
                # -----------------------------

                with st.expander(
                    "🔎 View retrieved document sections"
                ):

                    for i, chunk in enumerate(
                        relevant_chunks
                    ):

                        st.write(
                            f"**Retrieved Chunk {i + 1}**"
                        )

                        st.write(chunk)


            except Exception as e:

                error_message = str(e)

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):

                    st.warning(
                        "Gemini is temporarily busy. "
                        "Please try again."
                    )

                else:

                    st.error(
                        "Something went wrong while "
                        "communicating with Gemini."
                    )

                    st.write(
                        f"Error: {e}"
                    )
import os
import gradio as gr
import fitz  # PyMuPDF
import numpy as np
import faiss
import traceback

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================== CONFIG ===============================
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_KEY:
    print("❌ WARNING: GROQ_API_KEY is missing. Add it in HuggingFace → Settings → Secrets.")

client = Groq(api_key=GROQ_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =============================== PDF READER ===============================
def extract_text_from_pdf(pdf_file):
    """Extract all text from an uploaded PDF."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# =============================== TEXT CHUNKING ===============================
def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# =============================== FAISS VECTOR STORE ===============================
def build_faiss_index(chunks):
    """Create FAISS index from list of chunks."""
    embeddings = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, chunks, index, k=3):
    """Retrieve k most relevant chunks."""
    q_embed = embedder.encode([query]).astype("float32")
    distances, idx = index.search(q_embed, k)
    return [chunks[i] for i in idx[0]]


# =============================== GROQ LLM FUNCTIONS ===============================
def ask_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return traceback.format_exc()


def explain_topic(text):
    return ask_llm(f"Explain this in simple words:\n\n{text}")


def summarize_text(text):
    return ask_llm(f"Summarize the following text clearly:\n\n{text}")


def generate_mcqs(text):
    return ask_llm(
        f"Generate 5 high-quality MCQs with options and answer key based on this text:\n\n{text}"
    )


# =============================== RAG PIPELINE ===============================
def rag_answer(pdf_file, question):
    try:
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(text)
        index = build_faiss_index(chunks)
        retrieved = retrieve(question, chunks, index, k=3)
        context = "\n\n".join(retrieved)

        prompt = f"""
Use ONLY the context below to answer the question. 
If the answer is not present, reply:
"The document does not contain this information."
Context:
{context}
Question:
{question}
"""

        return ask_llm(prompt)

    except Exception:
        return traceback.format_exc()


# =============================== POLISHED UI ===============================

with gr.Blocks(theme=gr.themes.Soft(), title="EduBot+") as app:

    # HEADER
    gr.Markdown(
        """
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color:#2d6cdf; font-size: 42px;'>🎓 EduBot+ — AI Study Assistant</h1>
            <p style='font-size:18px; color:#444;'>
                Explain Topics • Summarize PDFs • Generate MCQs • Ask Questions (RAG)
            </p>
            <hr style='border:1px solid #ddd; margin-top:20px;'>
        </div>
        """
    )

    # ------------------------ EXPLAIN TAB ------------------------
    with gr.Tab("Explain"):
        gr.Markdown("### 🧠 Explain any topic clearly")

        inp = gr.Textbox(
            label="Enter topic or text",
            placeholder="e.g., Explain the Krebs cycle",
            lines=3
        )
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="Explanation", lines=10, show_copy_button=True)

        gr.Button("Explain", variant="primary").click(
            lambda x, f: explain_topic(extract_text_from_pdf(f) if f else x),
            inputs=[inp, pdf],
            outputs=out
        )

    # ------------------------ SUMMARIZE TAB ------------------------
    with gr.Tab("Summarize"):
        gr.Markdown("### ✂️ Summarize long text or PDFs")

        inp = gr.Textbox(
            label="Enter long text",
            placeholder="Paste long text or upload a PDF…",
            lines=8
        )
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="Summary", lines=10, show_copy_button=True)

        gr.Button("Summarize", variant="primary").click(
            lambda x, f: summarize_text(extract_text_from_pdf(f) if f else x),
            inputs=[inp, pdf],
            outputs=out
        )

    # ------------------------ MCQ TAB ------------------------
    with gr.Tab("Generate MCQs"):
        gr.Markdown("### ❓ Generate exam-style MCQs")

        inp = gr.Textbox(
            label="Enter text",
            placeholder="Paste lecture notes, topic list, or syllabus…",
            lines=8
        )
        pdf = gr.File(label="Upload PDF (optional)")
        out = gr.Textbox(label="MCQs", lines=12, show_copy_button=True)

        gr.Button("Generate MCQs", variant="primary").click(
            lambda x, f: generate_mcqs(extract_text_from_pdf(f) if f else x),
            inputs=[inp, pdf],
            outputs=out
        )

    # ------------------------ RAG TAB ------------------------
    with gr.Tab("Ask PDF (RAG)"):
        gr.Markdown("### 📄 Ask questions from your PDF")

        pdf_file = gr.File(label="Upload PDF")
        question = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is the conclusion of this study?",
            lines=2
        )
        answer = gr.Textbox(label="RAG Answer", lines=14, show_copy_button=True)

        gr.Button("Ask", variant="primary").click(
            rag_answer,
            inputs=[pdf_file, question],
            outputs=answer
        )


# Launch app
app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))



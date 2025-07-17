import os
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_KEY")

st.set_page_config(page_title="Batch RAG QA Evaluation")
st.title("Batch RAG Evaluation on Uploaded Questions")

# Sidebar: Upload PDF, CSV, and select Groq LLM model
with st.sidebar:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    csv_file = st.file_uploader("Upload QA CSV", type=["csv"])
    k = st.slider("Top-k (retriever)", 1, 10, 3)

    # Model selection for Groq
    groq_models = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "mixtral-8x7b-instruct",
    "qwen2-72b-instruct",
    "qwen2-7b-instruct",
    "yi-34b-chat"
    ]
    selected_model  = st.selectbox("Select Groq Model", groq_models)

if not pdf_file or not csv_file:
    st.info("Upload both a PDF and your QA CSV to run batch evaluation.")
    st.stop()

# --- PDF Processing (single load for all queries) ---
with st.spinner("Processing PDF..."):
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(pdf_file.read())
    temp_pdf.flush()

    pages = PyPDFLoader(temp_pdf.name).load()
    splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, hf_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

# --- CSV Processing ---
qa_df = pd.read_csv(csv_file)
for col in ["Relevant Chunk IDs", "Reference Answer"]:
    if col not in qa_df.columns:
        st.error("CSV missing one of the required columns: 'Relevant Chunk IDs', 'Reference Answer'.")
        st.stop()
qa_df = qa_df.fillna("")

# --- Metric Functions ---
def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant)) if relevant else 0.0
def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k
def mrr(retrieved, relevant):
    return next((1/i for i, r in enumerate(retrieved, 1) if r in relevant), 0.0)
def answer_relevance(q, a):
    qv, av = hf_embeddings.embed_query(q), hf_embeddings.embed_query(a)
    return float(cosine_similarity([qv],[av])[0][0])
def bertscore(a, ref):
    P, R, F1 = bert_score([a], [ref], lang="en", model_type="bert-base-uncased")
    return float(P[0]), float(R[0]), float(F1[0])

# --- Run Batch Evaluation for Selected Model ---
def run_batch_evaluation(llm_model_name):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model_name)
    results = []
    for idx, row in qa_df.iterrows():
        q_text = row["Question"]
        gt_chunk_str = str(row["Relevant Chunk IDs"])
        reference = str(row["Reference Answer"]).strip()
        relevant_ids = [int(i.strip()) for i in re.split(r"[,\s]+", gt_chunk_str) if i.strip().isdigit()]

        docs = retriever.invoke(q_text)
        retrieved_ids, retrieved_docs = [], []
        for doc in docs:
            match = [i for i, c in enumerate(chunks) if c.page_content == doc.page_content]
            if match:
                retrieved_ids.append(match[0])
                retrieved_docs.append(doc)
        context = "\n\n".join([d.page_content for d in retrieved_docs])

        prompt = f"Use only this context to answer:\n\n{context}\n\nQ: {q_text}\nA:"
        model_resp = llm.invoke(prompt)
        answer = model_resp.content if hasattr(model_resp, "content") else str(model_resp)
        answer = answer.strip()

        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        mrr_val = mrr(retrieved_ids, relevant_ids)
        cos_sim = answer_relevance(q_text, answer)
        bertP, bertR, bertF1 = bertscore(answer, reference) if reference else (None, None, None)

        results.append({
            "Q#": idx+1,
            "Question": q_text,
            "Reference Answer": reference,
            "Generated Answer": answer,
            "Ground-truth Chunk IDs": relevant_ids,
            "Retrieved Chunk IDs": retrieved_ids,
            "Recall@k": r_at_k,
            "Precision@k": p_at_k,
            "MRR": mrr_val,
            "Cosine Similarity (Q↔A)": cos_sim,
            "BERTScore (P)": bertP,
            "BERTScore (R)": bertR,
            "BERTScore (F1)": bertF1
        })
    return pd.DataFrame(results)

# Run evaluation for selected model
results_df = run_batch_evaluation(selected_model)

# --- Show Data Table ---
st.subheader(f"Evaluation Metrics for All Questions ({selected_model})")
st.dataframe(results_df[
    ["Q#", "Recall@k","Precision@k","MRR","Cosine Similarity (Q↔A)","BERTScore (F1)"]
].style.format("{:.3f}", subset=["Recall@k","Precision@k","MRR","Cosine Similarity (Q↔A)","BERTScore (F1)"]))

# --- Plot Metrics ---
metric_names = ["Recall@k", "Precision@k", "MRR", "Cosine Similarity (Q↔A)", "BERTScore (F1)"]
for metric in metric_names:
    fig = px.bar(results_df, x="Q#", y=metric, text_auto='.2f', title=f"{metric} Across All Questions — {selected_model}", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# --- Summary Report: Show Average Across All Questions ---
averages = results_df[metric_names].mean().to_frame(name="Average Score")
st.subheader("Summary Report: Model Averages (Across All Questions)")
st.dataframe(averages.T.style.format("{:.3f}"))

# --- Interactive Inspection ---
selected_idx = st.selectbox("Inspect a question:", options=results_df.index, format_func=lambda i: f"Q{results_df.loc[i,'Q#']}: {results_df.loc[i,'Question'][:60]}...")
if selected_idx is not None:
    row = results_df.iloc[selected_idx]
    st.markdown(f"### Question {row['Q#']}")
    st.markdown(f"**Question text:** {row['Question']}")
    st.markdown(f"**Reference Answer:** {row['Reference Answer']}")
    st.markdown(f"**Generated Answer:** {row['Generated Answer']}")
    st.markdown(f"**Ground-truth Chunk IDs:** {row['Ground-truth Chunk IDs']}")
    st.markdown(f"**Retrieved Chunk IDs:** {row['Retrieved Chunk IDs']}")
    with st.expander("Show Text of Each Retrieved Chunk"):
        for cid in row['Retrieved Chunk IDs']:
            st.markdown(f"**Chunk {cid}:**")
            st.write(chunks[cid].page_content)

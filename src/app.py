"""Streamlit demo: ask questions, watch self-healing events live.

Run:  streamlit run src/app.py
"""
import streamlit as st
from graph import ask

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹")
st.title("🩹 Self-Healing RAG")
st.caption("A RAG pipeline that grades its own retrieval & answers — and repairs itself when they fail.")

question = st.text_input("Ask a question about your documents:")

if st.button("Ask") and question:
    with st.spinner("retrieve → grade → (heal?) → generate → grade ..."):
        result = ask(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Self-healing trace")
    if result["healing_log"]:
        for event in result["healing_log"]:
            (st.warning if event.startswith("[DETECT]") else st.info)(event)
    else:
        st.success("✅ Passed all graders on the first attempt — no healing needed.")

    with st.expander("Retrieved chunks"):
        for i, c in enumerate(result["chunks"], 1):
            st.text(f"--- chunk {i} ---\n{c[:500]}")

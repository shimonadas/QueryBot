import streamlit as st
import os
import tempfile

from ingest import extract_text_from_pdf, chunk_text, embed_chunks, save_to_faiss
from query_engine import load_vector_index, retrieve_top_chunks, build_prompt, ask_ollama


# initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False


# configure page and header
st.set_page_config(page_title="QueryBot", layout="centered")
st.title("🤖 QueryBot – chat with your PDF")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .chat-bubble {
        background-color: #f1f3f4;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .user-bubble {
        background-color: #e0f7fa;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
Upload a PDF like the *2025 Global F1 Fan Survey*, ask questions, and 
get insights powered by a local LLM (Mistral).
""")

# file uploader section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if a file is uploaded, process it
if uploaded_file is not None:
    with st.spinner("processing document..."):
        # save the uploaded file to a temp location
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # run the document ingestion pipeline
        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text)
        nodes, embed_model = embed_chunks(chunks)
        save_to_faiss(nodes, embed_model)

        st.session_state.index_loaded = True
        st.success("document processed!")


# chatbot interface
if st.session_state.index_loaded:
    user_input = st.text_input("Ask a question about your PDF:")

    # if user enters a question, run query pipeline
    if user_input:
        with st.spinner("thinking..."):
            # load index from disk
            @st.cache_resource
            def get_index():
                return load_vector_index()

            index = get_index()

            # retrieve relevant chunks from vector DB
            raw_response = retrieve_top_chunks(index, user_input)
            context_texts = [str(node.node.get_content()) for node in raw_response.source_nodes]

            # build prompt and call LLM
            prompt = build_prompt(user_input, context_texts)
            answer = ask_ollama(prompt)

            # store Q&A in session history
            st.session_state.chat_history.append(("🧑 you", user_input))

            # create source citation block
            sources = "\n\n**Sources:**\n"
            for i, chunk in enumerate(context_texts):
                sources += f"\n> *Chunk {i+1}:* {chunk[:300]}..."
            # append bot answer + citations
            st.session_state.chat_history.append(("🤖 QueryBot", answer + sources))

    # display chat history
    for role, msg in st.session_state.chat_history:
        if role == "🧑 you":
            st.markdown(f"<div style='margin-bottom:10px'><b>{role}:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#f5f5f5; padding:10px; border-radius:10px; margin-bottom:20px'><b>{role}:</b><br>{msg}</div>", unsafe_allow_html=True)

else:
    st.info("please upload a PDF to begin.")

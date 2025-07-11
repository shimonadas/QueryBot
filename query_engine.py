import requests
from llama_index.core import Settings, VectorStoreIndex, StorageContext, \
    load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# prevent OpenAI LLM from being loaded
Settings.llm = None


def load_vector_index(persist_dir: str = "vector_store") -> BaseIndex:
    """
    Load the FAISS-backed index that `ingest.py` wrote.
    """
    embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        vector_store=vector_store,
    )
    return load_index_from_storage(storage_context, embed_model=embed_model)


def retrieve_top_chunks(index, user_query, top_k=3):
    """
    Retrieve top_k relevant document chunks based on semantic similarity.
    """
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(user_query)
    return response


# def build_prompt(user_question, context_texts):
#     """
#     Combine retrieved chunks and user query into a prompt for the LLM.
#     """
#     context = "\n\n---\n\n".join(context_texts)
#     prompt = f"""You are a helpful assistant. Use the context below to answer the question truthfully.
#
# Context:
# {context}
#
# Question: {user_question}
#
# Answer:"""
#     return prompt

def build_prompt(user_question, context_texts):
    context = "\n\n---\n\n".join(context_texts)

    with open("prompts/prompt_template.txt", "r") as f:
        template = f.read()

    prompt = template.replace("{context}", context).replace("{question}", user_question)
    return prompt


def ask_ollama(prompt, model="mistral"):
    """
    Send the prompt to a local Ollama model and return the response.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    return data["response"]


def main():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    index = load_vector_index()
    while True:
        user_question = input("\nAsk a question about the F1 PDF (or type 'exit'): ")
        if user_question.lower() == "exit":
            break

        print("🔍 Retrieving relevant content...")
        raw_response = retrieve_top_chunks(index, user_question)
        context_texts = [str(node.node.get_content()) for node in raw_response.source_nodes]

        print("🧠 Generating answer using Mistral...")
        prompt = build_prompt(user_question, context_texts)
        answer = ask_ollama(prompt)

        print("\n🤖 Answer:\n", answer)


if __name__ == "__main__":
    main()

# importing the libraries
import fitz  # PyMuPDF
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex  # for embeddings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
import faiss


# extracting the text from the document
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()  # ignore error, pycharm overly cautious
    return full_text


# turning the text into 500 word chunks
def chunk_text(text):
    overlap = 50
    chunk_size = 500
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks


# turning the text into vectors
def embed_chunks(chunks):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    nodes = [Document(text=chunk) for chunk in chunks]
    return nodes, embed_model


# saving the embeddings to FAISS which searches for similar vectors
def save_to_faiss(nodes, embed_model, output_dir="vector_store"):
    """
    Build a FAISS index, embed the nodes, and persist to disk.
    """
    # building an empty FAISS index of the correct dim
    dim = len(embed_model.get_text_embedding("dimension check"))
    faiss_index = faiss.IndexFlatL2(dim)
    # wrapping it in the LlamaIndex adapter
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # tell LlamaIndex to use this store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,   # omit if you set Settings.embed_model
    )
    # persist everything
    storage_context.persist(persist_dir=output_dir)
    print(f"✅ Vector index saved to: {output_dir}")


# running the full ingestion pipeline
def main():
    pdf_path = "data/2025_Global_F1_Fan_Survey_Final.pdf"
    text = extract_text_from_pdf(pdf_path)
    print("✅ PDF loaded.")
    chunks = chunk_text(text)
    print(f"✅ Text split into {len(chunks)} chunks.")
    nodes, embed_model = embed_chunks(chunks)
    print("✅ Embeddings created.")
    save_to_faiss(nodes, embed_model)
    print("✅ Ingestion complete!")


if __name__ == "__main__":
    main()


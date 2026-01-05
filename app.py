from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.chat_engine.types import ChatMode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SimilarityPostprocessor

from src.pdf_parser import ReaderType, PDFReaderFactory
from src.file_manager import FileManager

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONTENT_DIR = Path("./uploaded_documents/content")
NAME_FILE = Path("./uploaded_documents/names.pkl")
HASH_FILE = Path("./uploaded_documents/hashes.pkl")
file_manager = FileManager(persistent_path=CONTENT_DIR,
                           file_names=NAME_FILE, file_hashes=HASH_FILE)

VECTOR_DB_PATH = Path("./chroma_db")
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# AI models
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="auto",
    timeout=60
)

# PDF parsers
pdf_reader_factory = PDFReaderFactory()

# Single chat memory for all conversations
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000
)

# Similarity threshold for filtering retrieved nodes
SIMILARITY_THRESHOLD = 0.5
similarity_postprocessor = SimilarityPostprocessor(
    similarity_cutoff=SIMILARITY_THRESHOLD
)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Document Q&A API is running"}

@app.get("/readers")
def list_readers():
    """
    List available PDF readers
    """
    return {
        "readers": pdf_reader_factory.get_all_readers_info(),
        "default": pdf_reader_factory.get_default_reader()
    }

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    reader: ReaderType = Form(ReaderType.DOCLING)
):
    """
    Upload a document, parse it, chunk it, embed it, and store in vector DB

    Args:
        file: The PDF file to upload
        reader: The reader to use for parsing (pdf_reader, docling, or llama_parse)
    """
    try:
        file_manager.add_to_buffer(file)
        file_manager.validate_file(file.filename)
    except HTTPException as e:
        return {
            "success": False,
            "message": e,
            "filename": file.filename,
        }
    file_manager.store(file)

    # Get the appropriate reader from the factory
    try:
        selected_reader = pdf_reader_factory.get_reader(reader)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    documents = selected_reader.load_data(str(CONTENT_DIR / file.filename)) # type: ignore
    for doc in documents:
        doc.metadata["filename"] = file.filename

    # Get the count before indexing
    count_before = chroma_collection.count()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # Get the count after indexing to determine how many chunks were added
    count_after = chroma_collection.count()
    num_chunks = count_after - count_before

    return {
        "message": "Document uploaded and processed successfully",
        "filename": file.filename,
        "reader_used": reader.value,
        "vectors_created": num_chunks,
        "duplicate": False
    }

@app.get("/documents")
def list_documents():
    """
    List all uploaded documents
    """
    return file_manager.list_documents()

@app.post("/query")
def query_documents(request: QueryRequest):
    """
    Query the documents and return answer with citations.
    Maintains conversation history across all queries.
    Filters results using similarity threshold to improve answer quality.
    """
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Use chat engine instead of query engine for memory support
        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=3,
            chat_mode=ChatMode.CONTEXT,  # Uses retrieved context + chat history
            memory=chat_memory,
            node_postprocessors=[similarity_postprocessor],
            system_prompt="You are a helpful AI assistant that answers questions based on the provided documents. Use the context from the documents to provide accurate answers. Provide concise answers."
        )

        response = chat_engine.chat(request.question)

        # Extract source information
        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "Unknown"),
                "text": node.text,
                "score": node.score
            })

        # Check if no sources passed the similarity threshold
        if len(sources) == 0:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question. This could mean the question is not related to the uploaded documents.",
                "sources": sources,
                "sources_count": 0,
                "warning": "No sources met the similarity threshold"
            }

        return {
            "answer": str(response),
            "sources": sources,
            "sources_count": len(sources)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.delete("/chat")
def clear_chat_history():
    """
    Clear chat conversation history
    """
    global chat_memory
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000
    )
    return {"message": "Chat history cleared successfully"}

@app.delete("/clear")
def clear_database():
    """
    Clear the vector database, delete all uploaded files, and clear chat history
    """
    global chroma_collection, vector_store, storage_context, chat_memory

    try:
        collection_count = chroma_collection.count()
        chroma_client.delete_collection("documents")

        chroma_collection = chroma_client.get_or_create_collection("documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        files_deleted = file_manager.clear_files()

        # Clear chat history
        chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000
        )

        return {
            "message": "Database, files, and chat history cleared successfully",
            "vectors_cleared": collection_count,
            "files_deleted": files_deleted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

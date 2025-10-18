from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import PDFReader
from llama_parse import LlamaParse
import shutil
from pathlib import Path
import uuid
from dotenv import load_dotenv

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

UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="auto",
    timeout=60
)

pdf_reader = PDFReader()
docling_reader = DoclingReader()
llama_parse_reader = LlamaParse()

class ReaderType(str, Enum):
    PDF_READER = "pdf_reader"
    DOCLING = "docling"
    LLAMA_PARSE = "llama_parse"

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
        "readers": [
            {
                "value": ReaderType.PDF_READER.value,
                "name": "PDF Reader",
                "description": "Basic PDF reader from LlamaIndex"
            },
            {
                "value": ReaderType.DOCLING.value,
                "name": "Docling",
                "description": "Advanced PDF reader with layout preservation"
            },
            {
                "value": ReaderType.LLAMA_PARSE.value,
                "name": "LlamaParse",
                "description": "LlamaIndex's premium parsing service"
            }
        ],
        "default": ReaderType.DOCLING.value
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
        file_extension = Path(file.filename).suffix.lower()
        if file_extension != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        file_id = str(uuid.uuid4())
        unique_filename = f"{file_id}_{file.filename}"
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Select the appropriate reader based on the parameter
        if reader == ReaderType.PDF_READER:
            selected_reader = pdf_reader
        elif reader == ReaderType.DOCLING:
            selected_reader = docling_reader
        elif reader == ReaderType.LLAMA_PARSE:
            selected_reader = llama_parse_reader
        else:
            raise HTTPException(status_code=400, detail=f"Invalid reader type: {reader}")

        documents = selected_reader.load_data(str(file_path))
        for doc in documents:
            doc.metadata["filename"] = file.filename
            doc.metadata["file_id"] = file_id

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
            "file_id": file_id,
            "reader_used": reader.value,
            "vectors_created": num_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/documents")
def list_documents():
    """
    List all uploaded documents
    """
    documents = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            # Extract original filename (remove UUID prefix)
            parts = file_path.name.split("_", 1)
            if len(parts) == 2:
                file_id, original_name = parts
                documents.append({
                    "file_id": file_id,
                    "filename": original_name,
                    "path": str(file_path)
                })
    return {"documents": documents}

@app.post("/query")
def query_documents(request: QueryRequest):
    """
    Query the documents and return answer with citations
    """
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            response_mode="compact"
        )

        response = query_engine.query(request.question)

        # Extract source information
        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "Unknown"),
                "text": node.text,
                "score": node.score
            })

        return {
            "answer": str(response),
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.delete("/clear")
def clear_database():
    """
    Clear the vector database and delete all uploaded files
    """
    global chroma_collection, vector_store, storage_context

    try:
        collection_count = chroma_collection.count()
        chroma_client.delete_collection("documents")

        chroma_collection = chroma_client.get_or_create_collection("documents")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        files_deleted = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                files_deleted += 1

        return {
            "message": "Database and files cleared successfully",
            "vectors_cleared": collection_count,
            "files_deleted": files_deleted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

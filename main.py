from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import shutil
from pathlib import Path
import uuid

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

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = Ollama(model="llama3.2", request_timeout=120.0)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

reader = DoclingReader()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Document Q&A API is running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, parse it, chunk it, and store in vector DB
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
        
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)],
            file_extractor={".pdf": reader}
        ).load_data()
        
        for doc in documents:
            doc.metadata["filename"] = file.filename
            doc.metadata["file_id"] = file_id
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "file_id": file_id,
            "chunks_created": len(documents)
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
        
        # Create query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,  # Retrieve top 3 most relevant chunks
            response_mode="compact"
        )
        
        # Query the documents
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

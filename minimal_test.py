import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import PDFReader
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()

pdf_path = "tests/inputs/doclaynet.pdf"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#llm = Ollama(model="llama3.2", request_timeout=120.0)
llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    provider="auto"
)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#reader = DoclingReader()
#reader = LlamaParse()
reader = PDFReader()
documents = reader.load_data(pdf_path)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

query_engine = index.as_query_engine(llm=llm, similarity_top_k=1)
query = "What is DocLayNet?"
response = query_engine.query(query)
print(f"\nAnswer: {response.response}")

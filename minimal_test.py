"""Minimal test script"""
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader

pdf_path = "tests/inputs/doclaynet.pdf"

# Setup
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=120.0)

# Alternative: Add performance options for llama3.2
# llm = Ollama(
#     model="llama3.2",
#     request_timeout=120.0,
#     additional_kwargs={
#         "num_predict": 256,  # Limit response length
#         "temperature": 0.7,
#     }
# )

# Create ephemeral DB
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load
reader = DoclingReader()
documents = reader.load_data(pdf_path)

print(f"DoclingReader loaded {len(documents)} documents")
print(f"Document length: {len(documents[0].text)} chars")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print(f"Docstore has {len(index.docstore.docs)} nodes")
print(f"ChromaDB collection has {chroma_collection.count()} nodes")

print("\n" + "="*80)
print("DIAGNOSTIC: Check node sizes")
print("="*80)
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("What is DocLayNet?")
print(f"\nTotal nodes retrieved: {len(nodes)}")
for i, node in enumerate(nodes):
    print(f"  Node {i+1}: {len(node.text)} chars, Score: {node.score:.4f}")

print("\n" + "="*80)
print("METHOD 1: Using Retriever (returns nodes with scores)")
print("="*80)
print(f"\nRetrieved {len(nodes)} nodes")
for i, node in enumerate(nodes):
    print(f"\n--- Node {i+1} (Score: {node.score:.4f}) ---")
    print(f"{node.text[:300]}...")

print("\n" + "="*80)
print("METHOD 2: Test Ollama directly (simple prompt)")
print("="*80)
import time
test_prompt = "Say hello in one sentence."
print(f"Testing Ollama with simple prompt: '{test_prompt}'")
start = time.time()
try:
    simple_response = llm.complete(test_prompt)
    elapsed = time.time() - start
    print(f"✓ Ollama responded in {elapsed:.2f}s: {simple_response.text}")
except Exception as e:
    print(f"✗ Ollama test failed: {e}")

print("\n" + "="*80)
print("METHOD 3: Using Query Engine with timeout tracking")
print("="*80)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=1)
query = "What is DocLayNet?"
print(f"Query: '{query}'")
print(f"Context size being sent: ~{len(nodes[0].text)} chars")
print("Querying... (this may take a while)")
start = time.time()
try:
    response = query_engine.query(query)
    elapsed = time.time() - start
    print(f"✓ Query completed in {elapsed:.2f}s")
    print(f"\nAnswer: {response.response}")
    print(f"\nUsed {len(response.source_nodes)} source nodes")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ Query failed after {elapsed:.2f}s: {e}")


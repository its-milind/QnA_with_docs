import os
import tempfile
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# LangChain Loaders & Embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.chat_models import init_chat_model

#Pinecone Client
from pinecone import Pinecone

load_dotenv()

# Environment Variables
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

app = FastAPI(title="Multi-User Isolated RAG API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clients
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Serve Static Files and Root Route
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class TerminateRequest(BaseModel):
    session_id: str

def load_file(file_path: str, filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError('Unsupported format! Upload .txt, .docx, or .pdf')

# API Endpoints

@app.get("/api/status")
def get_status(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        return {"is_loaded": False, "filename": None, "chunk_count": 0, "page_count": 0}

    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        session_stats = namespaces.get(x_session_id, {})
        vector_count = session_stats.get("vector_count", 0)

        return {
            "is_loaded": vector_count > 0,
            "filename": "Active Session Document" if vector_count > 0 else None,
            "chunk_count": vector_count,
            "page_count": 0
        }
    except Exception:
        return {"is_loaded": False, "filename": None, "chunk_count": 0, "page_count": 0}

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None)
):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is missing.")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ['.pdf', '.txt', '.docx']:
        raise HTTPException(status_code=400, detail="Supported formats: PDF, TXT, DOCX")

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{x_session_id}_{filename}")

    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        docs = load_file(temp_path, filename)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # Generate Embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings_list = embedding_model.embed_documents(texts)

        # Upsert Vectors to Pinecone using Namespace Isolation
        vectors_to_upsert = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings_list)):
            vectors_to_upsert.append({
                "id": f"{x_session_id}-chunk-{i}",
                "values": vector,
                "metadata": {
                    "page_content": chunk.page_content,
                    "page": chunk.metadata.get("page", 0),
                    "filename": filename
                }
            })

        # Batch upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert, namespace=x_session_id)

        return {
            "message": f"Successfully loaded {filename}",
            "filename": filename,
            "pages": len(docs),
            "chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/query")
def ask_question(request: QueryRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is missing.")

    try:
        # Embed user question
        query_vector = embedding_model.embed_query(request.query)

        # Vector search directly on Pinecone index
        search_response = index.query(
            namespace=x_session_id,
            vector=query_vector,
            top_k=request.k,
            include_metadata=True
        )

        matches = search_response.get("matches", [])
        if not matches:
            raise HTTPException(status_code=400, detail="No document indexed for your session.")

        context_chunks = [match["metadata"]["page_content"] for match in matches]
        context = "\n\n".join(context_chunks)

        system_message = SystemMessage(
            content=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know and there is no information in the document.\n\n"
                f"Context:\n{context}"
            )
        )

        llm = init_chat_model(model='openai/gpt-oss-120b', model_provider='groq')
        response = llm.invoke([system_message, HumanMessage(content=request.query)])

        sources = [
            {
                "page_content": m["metadata"]["page_content"],
                "metadata": {"page": m["metadata"].get("page", 0)}
            }
            for m in matches
        ]

        return {"query": request.query, "answer": response.content, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/terminate-session")
def terminate_session(payload: TerminateRequest):
    """Delete isolated namespace vector data on exit."""
    try:
        index.delete(delete_all=True, namespace=payload.session_id)
    except Exception:
        pass
    return {"status": "terminated"}

@app.delete("/api/clear")
def clear_document(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header is missing.")

    try:
        index.delete(delete_all=True, namespace=x_session_id)
        return {"message": "Active index cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

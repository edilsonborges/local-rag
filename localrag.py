from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import pypdf
import io
import os
import torch
import sqlite3
import numpy as np
import datetime
import requests
from typing import List, Optional
import asyncio
import httpx
import subprocess
import sys
import platform
import time
from dotenv import load_dotenv

load_dotenv()

# Database settings
DB_PATH = "localrag.db"

LLM_SERVER_HOST = os.getenv("LLM_SERVER_HOST", "localhost")
LLM_SERVER_PORT = int(os.getenv("LLM_SERVER_PORT", "1234"))
CHAT_API_URL = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/v1/chat/completions"

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    length_function=len,
)

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_db_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def array_to_sqlite(array):
    """Convert numpy array to SQLite-compatible format"""
    return json.dumps(array.tolist())

def sqlite_to_array(text):
    """Convert SQLite-stored text back to numpy array"""
    return np.array(json.loads(text))

def setup_database():
    """Set up the database and necessary tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("Setting up database...")
        
        # SQLite doesn't need extensions, but we'll get our embedding dimensions
        print("Checking embedding dimensions...")
        sample_text = "Sample text for dimension check"
        sample_embedding = model.encode(sample_text)
        embedding_dim = len(sample_embedding)
        
        # Create documents table if not exists
        print(f"Creating documents table for embeddings with {embedding_dim} dimensions...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT DEFAULT '{{}}'
            );
        """)
        
        print(f"✅ Database setup complete. Using embeddings with {embedding_dim} dimensions.")
        conn.commit()
        return True
    except Exception as e:
        print(f"❌ Database setup error: {str(e)}")
        print("Full error:", e.__class__.__name__, e)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def clear_database():
    """Clear all records from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM documents")
        
        conn.commit()
        print("✅ Database cleared successfully")
        return True
    except Exception as e:
        print(f"❌ Error clearing database: {str(e)}")
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    with open(pdf_path, "rb") as f:
        pdf_reader = pypdf.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_markdown(md_path):
    """Extract text from a Markdown file"""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

def store_documents_in_db(documents, embeddings, source_file=None):
    """Store documents and their embeddings in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert documents and embeddings
        for doc in documents:
            doc_id = doc["id"]
            content = doc["text"]
            embedding = embeddings[doc_id].cpu().numpy()
            
            # Convert embedding to SQLite-compatible format
            embedding_json = array_to_sqlite(embedding)
            
            # Create metadata
            metadata = {
                "source": source_file if source_file else "unknown",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Insert or update the document
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, content, embedding, metadata)
                VALUES (?, ?, ?, ?)
            """, (doc_id, content, embedding_json, json.dumps(metadata)))
        
        conn.commit()
        print(f"✅ Stored {len(documents)} documents in the database")
        return True
    except Exception as e:
        print(f"❌ Error storing documents in database: {str(e)}")
        print("Full error:", e.__class__.__name__, e)
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def load_documents_from_db():
    """Load all documents from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, content FROM documents")
        rows = cursor.fetchall()
        
        documents = [{"id": row[0], "text": row[1]} for row in rows]
        print(f"✅ Loaded {len(documents)} documents from the database")
        return documents
    except Exception as e:
        print(f"❌ Error loading documents from database: {e}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def process_file(file_path):
    """Process file (PDF or Markdown) and split into chunks"""
    # Determine file type and extract text
    if file_path.lower().endswith('.pdf'):
        print(f"Extracting text from PDF: {file_path}...")
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.md', '.markdown')):
        print(f"Extracting text from Markdown: {file_path}...")
        text = extract_text_from_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    print(f"Extracted {len(text)} characters of text")
    
    # Split text into chunks
    print("Splitting text into chunks...")
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Create documents with unique IDs
    documents = [{"id": str(uuid.uuid4()), "text": chunk} for chunk in chunks]
    
    # Generate embeddings for each document
    print("Generating embeddings...")
    doc_embeddings = {
        doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents
    }
    print("Embeddings generated")
    
    # Store documents and embeddings in the database
    store_documents_in_db(documents, doc_embeddings, file_path)
    
    return documents, doc_embeddings

def vector_search_in_db(query_embedding, limit=5):
    """Search for similar documents in the database using vector similarity"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if we have any documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        print(f"Found {count} total documents in database")
        
        if count == 0:
            print("⚠️ No documents in database to search")
            return None, 0
        
        # Convert query embedding to numpy array
        query_embedding_np = query_embedding.cpu().numpy()
        
        # Since SQLite doesn't have vector operations, we'll need to manually calculate similarities
        cursor.execute("SELECT id, content, embedding, metadata FROM documents")
        results = cursor.fetchall()
        
        similarities = []
        for row in results:
            doc_id = row[0]
            content = row[1]
            embedding = sqlite_to_array(row[2])
            metadata = json.loads(row[3])
            
            # Calculate cosine similarity
            similarity = 1 - np.linalg.norm(embedding - query_embedding_np)
            
            similarities.append((doc_id, content, metadata, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[3], reverse=True)
        
        # Take top results
        top_results = similarities[:limit]
        
        if not top_results:
            return None, 0
        
        # Get the best match
        best_doc_id, best_content, metadata, best_score = top_results[0]
        best_doc = {
            "id": best_doc_id, 
            "text": best_content,
            "metadata": metadata
        }
        
        return best_doc, float(best_score)
    except Exception as e:
        print(f"❌ Error searching documents in database: {str(e)}")
        print("Full error:", e.__class__.__name__, e)
        import traceback
        traceback.print_exc()
        return None, 0
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# FastAPI App
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Check if LLM server is running
def is_llm_server_running():
    """Check if the LLM server is running"""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Use environment variables for host and port
            return s.connect_ex((LLM_SERVER_HOST, LLM_SERVER_PORT)) == 0
    except Exception as e:
        print(f"Error checking LLM server status: {e}")
        return False

# Start LLM server if not running
def start_llm_server():
    """Start the LLM server using ollama"""
    if is_llm_server_running():
        print("LLM server is already running")
        return True
    
    try:
        # Check if ollama is installed
        if not os.path.exists(os.path.expanduser("~/.ollama")):
            print("Ollama not found. Installing...")
            # Installation depends on the OS
            if platform.system() == "Darwin":  # macOS
                os.system("curl -fsSL https://ollama.com/install.sh | sh")
            elif platform.system() == "Linux":
                os.system("curl -fsSL https://ollama.com/install.sh | sh")
            elif platform.system() == "Windows":
                print("Please install Ollama manually from https://ollama.com/download")
                return False
        
        # Pull the model if not already downloaded
        print("Checking for model hermes-3-llama-3.2-3b@q4_k_m...")
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "hermes-3-llama-3.2-3b" not in result.stdout:
            print("Downloading model hermes-3-llama-3.2-3b@q4_k_m...")
            subprocess.run(["ollama", "pull", "hermes-3-llama-3.2-3b@q4_k_m"])
        
        # Start the server
        print("Starting LLM server on port 1234...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the server to start
        for _ in range(10):
            if is_llm_server_running():
                print("LLM server started successfully")
                return True
            time.sleep(1)
        
        print("Failed to start LLM server")
        return False
    except Exception as e:
        print(f"Error starting LLM server: {e}")
        return False

@app.on_event("startup")
async def startup_db_client():
    """Initialize database on startup"""
    setup_database()
    # Try to start the LLM server
    start_llm_server()

@app.get("/")
async def get_chat_interface():
    """Serve the chat interface HTML"""
    return FileResponse("static/index.html")

@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF or Markdown) and process it"""
    try:
        content = await file.read()
        filename = file.filename
        
        # Check file type
        if not filename.lower().endswith(('.pdf', '.md', '.markdown')):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Only PDF and Markdown files are supported"
                }
            )
        
        # Save the file
        with open(f"temp_{filename}", "wb") as f:
            f.write(content)
        
        # Process the file
        documents, _ = process_file(f"temp_{filename}")
        
        # Clean up
        os.remove(f"temp_{filename}")
        
        return {
            "status": "success",
            "message": f"File {filename} processed successfully",
            "chunks": len(documents)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/clear-database/")
async def clear_db():
    """Clear all records from the database"""
    success = clear_database()
    if success:
        return {"status": "success", "message": "Database cleared successfully"}
    else:
        return {"status": "error", "message": "Failed to clear database"}

class QueryRequest(BaseModel):
    query: str

@app.post("/query-pdf/")
def query_pdf(request: QueryRequest):
    """Query the processed PDF documents"""
    # Encode the query
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    
    # Search in the database
    best_doc, score = vector_search_in_db(query_embedding)
    
    if not best_doc:
        return {"error": "No documents found in the database"}
    
    return {
        "document": best_doc.get("text"),
        "metadata": best_doc.get("metadata", {}),
        "score": score
    }

# Add new models for the chat completion request
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = -1
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    context: Optional[str]
    metadata: Optional[dict]
    score: Optional[float]

async def stream_ai_response(messages: List[Message], temperature: float = 0.0, max_tokens: int = -1, source_ref=None):
    """Stream response from the AI model"""
    try:
        payload = {
            "model": "hermes-3-llama-3.2-3b@q4_k_m",
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        # Send header with source reference information
        if source_ref:
            source_data = {
                "event": "source",
                "data": source_ref
            }
            yield f"data: {json.dumps(source_data)}\n\n"
        
        async with httpx.AsyncClient() as client:
            async with client.stream('POST', CHAT_API_URL, json=payload, headers={"Content-Type": "application/json"}) as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if data.get('choices') and len(data['choices']) > 0:
                                content = data['choices'][0].get('delta', {}).get('content', '')
                                if content:
                                    yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
                    elif line.strip() == "data: [DONE]":
                        break
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/chat/stream/")
async def chat_with_context_stream(request: ChatRequest):
    """Streaming chat endpoint that combines RAG with AI agent"""
    try:
        # Make sure LLM server is running
        if not is_llm_server_running():
            start_llm_server()
        
        # First, get relevant context from the database
        query_embedding = model.encode(request.query, convert_to_tensor=True)
        best_doc, score = vector_search_in_db(query_embedding)
        
        if not best_doc:
            return StreamingResponse(
                stream_ai_response([
                    Message(
                        role="assistant",
                        content="Desculpe, não encontrei informações relevantes na base de dados para responder sua pergunta."
                    )
                ]),
                media_type="text/event-stream"
            )
        
        # Prepare source reference information for the client
        source_ref = {
            "source": best_doc['metadata'].get('source', 'unknown'),
            "context": best_doc['text'],
            "score": float(score)
        }
        
        # Create messages for the AI agent
        messages = [
            Message(
                role="system",
                # content="Você é um assistente de defesa agropecuária. Responda perguntas sobre a Agrodefesa (Agência Goiana de Defesa Agropecuária) e sobre o sistema de defesa agropecuária do estado de Goiás (SIDAGO). Você responderá apenas em português brasileiro. Você deve responder de forma clara e objetiva. Todas as perguntas fora do contexto da Agrodefesa e do SIDAGO deverão ser respondidas reforçando o escopo de sua função, que é responder apenas perguntas sobre a Agrodefesa e o SIDAGO."
                content="Você responderá apenas em português brasileiro."
            ),
            Message(
                role="user",
                content=f"""Por favor, responda a seguinte pergunta usando o contexto fornecido.
                
Pergunta: {request.query}

Contexto relevante:
{best_doc['text']}

Responda de forma clara e direta, usando apenas as informações do contexto fornecido. Se o contexto não for suficiente para responder a pergunta completamente, indique isso na resposta."""
            )
        ]
        
        return StreamingResponse(
            stream_ai_response(
                messages, 
                temperature=request.temperature, 
                max_tokens=request.max_tokens,
                source_ref=source_ref
            ),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print("Full error:", e.__class__.__name__, e)
        import traceback
        traceback.print_exc()
        return StreamingResponse(
            stream_ai_response([
                Message(
                    role="assistant",
                    content="Desculpe, ocorreu um erro ao processar sua pergunta."
                )
            ]),
            media_type="text/event-stream"
        )

@app.get("/database-stats/")
async def get_database_stats():
    """Get statistics about the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get total number of documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_chunks = cursor.fetchone()[0]
        
        return {
            "status": "success",
            "total_chunks": total_chunks
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # Setup the database
    setup_success = setup_database()
    
    if not setup_success:
        print("❌ Failed to set up database. Exiting.")
        exit(1)
    
    # Check if we have documents in the database
    documents = load_documents_from_db()
    
    # Start LLM server if not already running
    start_llm_server()
    
    if documents:
        print(f"✅ Found {len(documents)} documents in the database")
        
        # Show some sample chunks
        print("\nSample chunks:")
        for i in range(min(3, len(documents))):
            print(f"\nChunk {i+1}:\n{documents[i]['text']}\n")
        
        # Test a sample query
        print("\nTesting sample queries using vector search...")
        sample_queries = [
            # "O que é o SIDAGO?",
            # "Como emitir uma GTA?",
            # "Quais são as regras para vacinação?"
        ]
        
        for query in sample_queries:
            query_embedding = model.encode(query, convert_to_tensor=True)
            best_doc, score = vector_search_in_db(query_embedding)
            if best_doc:
                print(f"\nQuery: {query}")
                print(f"Best match (score: {score:.4f}):")
                print(f"Source: {best_doc['metadata'].get('source', 'unknown')}")
                print(f"Content:\n{best_doc['text']}")
            else:
                print(f"\nNo results found for query: {query}")
    else:
        print("No documents found in database. Use the /upload-file/ endpoint to add documents.")
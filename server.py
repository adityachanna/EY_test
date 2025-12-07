"""
Pharma Analyst API Server
Exposes the PharmaAnalystAgent via FastAPI with support for:
- Synchronous queries
- Streaming responses (Server-Sent Events)
- Health checks
- Automatic API documentation (Swagger/OpenAPI)
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import json
import asyncio
from app import create_pharma_analyst, PharmaAnalystAgent

# Initialize FastAPI app
app = FastAPI(
    title="Pharma Analyst AI API",
    description="Strategic Pharmaceutical Research & Analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent (Singleton pattern for efficiency)
# We initialize it once so we don't reload environment/tools on every request
agent_instance: Optional[PharmaAnalystAgent] = None

def get_agent() -> PharmaAnalystAgent:
    global agent_instance
    if agent_instance is None:
        try:
            print("Initializing Pharma Analyst Agent...")
            agent_instance = create_pharma_analyst()
            print("Agent initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize agent: {e}")
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")
    return agent_instance

# --- Pydantic Models for Request/Response ---

class QueryRequest(BaseModel):
    query: str = Field(..., description="The research question or task for the analyst", example="What are the repurposing opportunities for Minocycline in neurological disorders?")
    stream: bool = Field(False, description="Whether to stream the response chunk by chunk")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation memory persistence")
    internal_document: Optional[str] = Field(None, description="Custom internal document text to analyze")

class QueryResponse(BaseModel):
    response: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and Agent are healthy."""
    return {
        "status": "healthy",
        "agent_initialized": agent_instance is not None
    }

@app.post("/query", response_model=QueryResponse, tags=["Analysis"])
async def query_agent(request: QueryRequest):
    """
    Submit a query to the Pharma Analyst Agent.
    
    If `stream` is false (default), returns the full JSON response after generation.
    If `stream` is true, use the `/stream` endpoint logic instead or set header Accept: text/event-stream.
    """
    if request.stream:
        # If user requests streaming via standard POST but expects direct stream, 
        # usually it's handled via a separate endpoint or accepted logic. 
        # For simplicity in this endpoint: suggest using /stream via client or redirect internally.
        # But we can just call the stream handler directly if we wanted to support both here.
        # Let's keep strict separation for clarity, or return a helpful error.
        raise HTTPException(status_code=400, detail="For streaming, please use the /chat/stream endpoint for SSE.")

    agent = get_agent()
    try:
        # Run in a threadpool since langchain might be synchronous blocking
        response_text = await asyncio.to_thread(agent.query, request.query, request.thread_id, request.internal_document)
        return {"response": response_text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream", tags=["Analysis"])
async def stream_agent(request: QueryRequest):
    """
    Stream the agent's response using Server-Sent Events (SSE).
    Useful for real-time UIs to show generation progress.
    """
    agent = get_agent()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Run the synchronous generator in a thread to not block the event loop
            # This is a bit tricky with generators, so we iterate synchronously 
            # but usually we want true async. 
            # Since app.py 'stream_query' is a synchronous generator, we iterate it directly.
            # Warning: This might block the event loop if not careful. 
            # For production, we'd wrap this properly or use LangChain's async methods.
            
            # Simple iteration for now (LangChain invoke is usually fast enough between tokens or we accept minor block)
            # A better way is using run_in_executor for the whole thing and using a queue, 
            # but app.py's stream_query yields chunks.
            
            iterator = agent.stream_query(request.query, request.thread_id, request.internal_document)
            
            for chunk in iterator:
                # Format as SSE
                # chunk is usually an AIMessageChunk or similar dict/object
                # Let's make sure we extract text safely
                content = ""
                if hasattr(chunk, 'content'):
                    content = chunk.content
                elif isinstance(chunk, dict) and 'messages' in chunk:
                     # If it's a state dump, we might ignore or extract
                     if chunk['messages']:
                        content = chunk['messages'][-1].content
                else:
                    content = str(chunk)

                if content:
                    # SSE format: data: <json>\n\n
                    data = json.dumps({"token": content})
                    yield f"data: {data}\n\n"
                    # Small yield to let loop breathe
                    await asyncio.sleep(0)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Clean output for user
    print("\nStarting Pharma Analyst API Server...")
    print("Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

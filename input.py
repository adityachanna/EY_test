"""
Internal Knowledge Agent Tool for App.py
This module defines the internal knowledge retrieval tool that accesses
company's internal documents via vector store (Pinecone + Google Embeddings).
"""

import os
import functools
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class KnowledgeInput(BaseModel):
    """Input schema for internal knowledge search."""
    query: str = Field(description="The query to search internal company knowledge base.")


@functools.lru_cache(maxsize=1)
def _get_vectorstore():
    """
    Lazy-load the VectorStore for internal knowledge.
    This prevents the app from crashing on import if keys are missing/invalid.
    Returns None if initialization fails.
    """
    try:
        print("Initializing Google Embeddings for Internal Knowledge...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("API_4")
        )
        print("Initializing Pinecone VectorStore for Internal Knowledge...")
        vectorstore = PineconeVectorStore(
            index_name="boundless-alder",
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        print("Internal Knowledge VectorStore initialized successfully.")
        return vectorstore
    except Exception as e:
        print(f"ERROR: Failed to initialize Internal Knowledge VectorStore: {e}")
        return None


@tool("internal_knowledge_search", args_schema=KnowledgeInput)
def internal_knowledge_tool(query: str) -> str:
    """
    Search the internal company knowledge base for relevant information.
    
    This tool retrieves information from the company's internal documents stored in a vector database.
    Use this for:
    - Company-specific policies, procedures, and guidelines
    - Internal research reports and white papers
    - Product documentation and technical specifications
    - Historical data about company products and their performance
    - Internal financial reports and earnings data
    - Proprietary knowledge not available in public sources
    
    The tool acts as a **Knowledge Agent** that intelligently retrieves contextually 
    relevant information from the company's internal document repository.
    
    Returns a string containing relevant internal documents with their sources and metadata.
    """
    vectorstore = _get_vectorstore()
    
    if vectorstore is None:
        return (
            "Error: Internal Knowledge Base is currently unavailable. "
            "This may be due to authentication or connection issues. "
            "Please contact the administrator or check system logs."
        )
    
    try:
        # Retrieve relevant documents from vector store
        retrieved_docs = vectorstore.similarity_search(query, k=4)
        
        if not retrieved_docs:
            return (
                "No relevant internal documents found for this query. "
                "The internal knowledge base may not contain information "
                "about this topic, or the query may need to be rephrased."
            )
        
        # Format results with sources and metadata
        formatted_results = []
        for doc in retrieved_docs:
            source_info = doc.metadata if doc.metadata else "Unknown Source"
            formatted_results.append(
                f"Source: {source_info}\nContent: {doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during internal knowledge retrieval: {str(e)}"


# Export the tool for use in other modules
__all__ = ['internal_knowledge_tool']

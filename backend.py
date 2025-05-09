import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Body, Header
from pydantic import BaseModel

# Import from your core_logic file
from llm_pipeline import (
    initialize_rag_components,
    get_rag_chain_response,
    convert_to_langchain_messages,
    convert_to_serializable_history,
    RETRIEVER_INSTANCE, # To check if initialized
    LLM_INSTANCE # To check if initialized
)
from langchain_core.messages import HumanMessage, AIMessage


# --- FastAPI App Initialization ---
app = FastAPI(
    title="OTMT-Pal API with Sessions",
    description="API for IIITD OTMT Chatbot with session management.",
    version="1.1.0"
)

# --- Session Management ---
chat_sessions: Dict[str, Dict[str, Any]] = {} # session_id -> {"messages": List[Dict], "last_interaction_time": datetime}
SESSION_TIMEOUT = timedelta(hours=24)

# --- Pydantic Models for API ---
class ChatMessageIO(BaseModel): # Renamed to avoid conflict if core_logic had one
    type: str # "human" or "ai"
    content: str

class ChatTurnRequest(BaseModel):
    question: str
    session_key: Optional[str] = None # Client can provide an existing session key

class ChatTurnResponse(BaseModel):
    answer: str
    session_key: str # Always return the key being used (new or existing)
    chat_history: List[ChatMessageIO]
    retrieved_sources: Optional[List[Dict[str, Any]]] = None

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("Starting up OTMT-Pal API with Sessions...")
    try:
        initialize_rag_components() # This loads models, processes docs etc.
        print("OTMT-Pal API core logic initialized and ready.")
    except Exception as e:
        print(f"FATAL: Error during startup initialization: {e}")
        # In a real production scenario, you might want the app to exit or enter a degraded state
        # For now, it will print the error, and endpoints will likely fail if LLM_INSTANCE isn't set
        import traceback
        traceback.print_exc()


# --- Helper function for session handling ---
def get_or_create_session(session_key: Optional[str]) -> tuple[str, List[Dict[str,str]]]:
    now = datetime.now()
    
    if session_key and session_key in chat_sessions:
        session_data = chat_sessions[session_key]
        # Check for expiry
        if now - session_data["last_interaction_time"] > SESSION_TIMEOUT:
            print(f"Session {session_key} expired. Creating new one.")
            # Expired, so treat as new
            new_key = str(uuid.uuid4())
            chat_sessions[new_key] = {"messages": [], "last_interaction_time": now}
            # Clean up old key
            del chat_sessions[session_key]
            return new_key, []
        else:
            # Valid, active session
            session_data["last_interaction_time"] = now # Update timestamp
            return session_key, session_data["messages"]
    else:
        # No key provided, or key not found - create new session
        new_key = str(uuid.uuid4())
        if session_key:
            print(f"Session key {session_key} not found. Creating new session {new_key}.")
        else:
            print(f"No session key provided. Creating new session {new_key}.")
        chat_sessions[new_key] = {"messages": [], "last_interaction_time": now}
        return new_key, []

# --- FastAPI Endpoints ---
@app.post("/chat/", response_model=ChatTurnResponse)
async def handle_chat_turn(request: ChatTurnRequest = Body(...)):
    if not LLM_INSTANCE: # Check if core components initialized
        raise HTTPException(status_code=503, detail="System is not ready or failed to initialize. Please try again later.")

    current_session_key, current_serializable_history = get_or_create_session(request.session_key)
    
    # Convert serializable history to LangChain messages for the RAG chain
    langchain_history = convert_to_langchain_messages(current_serializable_history)

    try:
        # Get response from RAG chain
        result = get_rag_chain_response(request.question, langchain_history)
        answer = result["answer"]

        # Update history with new turn
        updated_langchain_history = langchain_history + [
            HumanMessage(content=request.question),
            AIMessage(content=answer)
        ]
        
        # Store updated serializable history back in session
        chat_sessions[current_session_key]["messages"] = convert_to_serializable_history(updated_langchain_history)
        chat_sessions[current_session_key]["last_interaction_time"] = datetime.now()


        sources = []
        for doc in result.get("source_documents", []):
             sources.append({
                "page_content_preview": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })

        return ChatTurnResponse(
            answer=answer,
            session_key=current_session_key,
            chat_history=chat_sessions[current_session_key]["messages"], # Return the full updated history
            retrieved_sources=sources
        )
    except Exception as e:
        print(f"Error during chat processing for session {current_session_key}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/health")
async def health_check():
    if LLM_INSTANCE and RETRIEVER_INSTANCE: # Assuming RETRIEVER_INSTANCE is also set in core_logic
         return {"status": "ok", "message": "OTMT-Pal API is running and RAG components are initialized."}
    return {"status": "degraded", "message": "RAG components not fully initialized."}


# To run (from terminal in the directory containing backend.py and core_logic.py):
# uvicorn backend:app --reload --host 0.0.0.0 --port 8000
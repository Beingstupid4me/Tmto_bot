import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging # Added for consistency

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Import the module itself, and specific functions you need from it
import llm_tester # Import the module
from llm_tester import ( # Import necessary functions
    initialize_rag_components,
    get_rag_chain_response,
    convert_to_langchain_messages,
    convert_to_serializable_history
)
# DO NOT import LLM_INSTANCE or RETRIEVER_INSTANCE directly here:
# from llm_pipeline import LLM_INSTANCE # <--- REMOVE THIS LINE if present

from langchain_core.messages import HumanMessage, AIMessage

# --- FastAPI App Initialization ---
app = FastAPI(
    title="OTMT-Pal API with Sessions",
    description="API for IIITD OTMT Chatbot with session management.",
    version="1.1.0"
)

# --- Session Management ---
chat_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT = timedelta(hours=24)

# --- Pydantic Models for API ---
class ChatMessageIO(BaseModel):
    type: str
    content: str

class ChatTurnRequest(BaseModel):
    question: str
    session_key: Optional[str] = None

class ChatTurnResponse(BaseModel):
    answer: str
    session_key: str
    chat_history: List[ChatMessageIO]
    retrieved_sources: Optional[List[Dict[str, Any]]] = None

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    logging.info("Backend: Starting up OTMT-Pal API with Sessions...")
    try:
        initialize_rag_components() # This calls the function in llm_pipeline
        # It will set llm_pipeline.LLM_INSTANCE and llm_pipeline.RETRIEVER_INSTANCE
        logging.info("Backend: OTMT-Pal API core logic initialization appears complete based on llm_pipeline logs.")
    except Exception as e:
        logging.error(f"Backend: FATAL Error during startup initialization: {e}", exc_info=True)

# --- Helper function for session handling (remains the same) ---
def get_or_create_session(session_key: Optional[str]) -> tuple[str, List[Dict[str,str]]]:
    now = datetime.now()
    
    if session_key and session_key in chat_sessions:
        session_data = chat_sessions[session_key]
        if now - session_data["last_interaction_time"] > SESSION_TIMEOUT:
            logging.info(f"Backend: Session {session_key} expired. Creating new one.")
            new_key = str(uuid.uuid4())
            chat_sessions[new_key] = {"messages": [], "last_interaction_time": now}
            if session_key in chat_sessions: # Ensure deletion if key exists
                del chat_sessions[session_key]
            return new_key, []
        else:
            session_data["last_interaction_time"] = now
            return session_key, session_data["messages"]
    else:
        new_key = str(uuid.uuid4())
        if session_key:
            logging.info(f"Backend: Session key {session_key} not found. Creating new session {new_key}.")
        else:
            logging.info(f"Backend: No session key provided. Creating new session {new_key}.")
        chat_sessions[new_key] = {"messages": [], "last_interaction_time": now}
        return new_key, []

# --- FastAPI Endpoints ---
@app.post("/chat/", response_model=ChatTurnResponse)
async def handle_chat_turn(request: ChatTurnRequest = Body(...)):
    # Access LLM_INSTANCE through the llm_pipeline module
    if not llm_tester.LLM_INSTANCE or not llm_tester.RETRIEVER_INSTANCE:
        logging.warning("Backend: /chat/ called but llm_pipeline.LLM_INSTANCE or RETRIEVER_INSTANCE is not set.")
        raise HTTPException(status_code=503, detail="System is not ready or failed to initialize. Please try again later.")

    current_session_key, current_serializable_history = get_or_create_session(request.session_key)
    langchain_history = convert_to_langchain_messages(current_serializable_history)

    try:
        logging.info(f"Backend: Processing chat for session {current_session_key}, question: {request.question}")
        result = get_rag_chain_response(request.question, langchain_history) # This function is from llm_pipeline
        answer = result["answer"]

        updated_langchain_history = langchain_history + [
            HumanMessage(content=request.question),
            AIMessage(content=answer)
        ]
        
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
            chat_history=chat_sessions[current_session_key]["messages"],
            retrieved_sources=sources
        )
    except Exception as e:
        logging.error(f"Backend: Error during chat processing for session {current_session_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/health")
async def health_check():
    # Access globals through the llm_pipeline module
    if llm_tester.LLM_INSTANCE and llm_tester.RETRIEVER_INSTANCE:
         return {"status": "ok", "message": "OTMT-Pal API is running and RAG components (from llm_pipeline) are initialized."}
    logging.warning("Backend: /health check found RAG components (from llm_pipeline) not initialized.")
    return {"status": "degraded", "message": "RAG components (from llm_pipeline) not fully initialized."}

# (Uvicorn run command remains the same in your terminal)

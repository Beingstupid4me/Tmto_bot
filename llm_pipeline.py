import os
import glob
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
MODEL_PATH = "../Qwen3-8B"
DOCUMENTS_PATH = "./Documents"
CONTEXT_FILE_PATH = "./context.txt"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BM25_K = 5

# --- Global Instances (initialized once) ---
LLM_INSTANCE = None
RETRIEVER_INSTANCE = None
# SPLIT_DOCS_INSTANCE = None # Not strictly needed globally if retriever is main user

# --- Prompt Template ---
_template = """You are "OTMT-Pal", a helpful AI assistant for the Office of Technology Management and Transfer (OTMT) at IIIT-Delhi.
Your developer is [Your Name/Team Name - ensure this is in context.txt or updated here]. You are built using Langchain and a Qwen model.
IIIT-Delhi (Indraprastha Institute of Information Technology, Delhi) is a state university in Delhi, India.
TRL (Technology Readiness Level) assessment is a method for estimating the maturity of technologies.

Your primary purpose is to:
1. Provide information about IIIT-Delhi and OTMT.
2. Answer frequently asked questions regarding OTMT processes (e.g., meeting staff, filing patents, licensing technology).
3. Provide information about technologies developed at IIIT-Delhi, based on the documents provided.

You MUST use the provided "Context" (retrieved documents) to answer the "Question".
If the Context does not contain the answer, or if the question is outside your scope, you MUST state that you cannot answer.
Do not make up information. If you don't know, say "I don't have enough information to answer that."

Examples of out-of-scope questions:
- "Can you give me the code for a graph?"
- "What's a good pizza recipe?"
- "Tell me about quantum physics (unless there's a specific IIITD technology on it in the Context)."

When asked an out-of-scope question, respond politely, for example: "I can help with information about IIITD's OTMT and its technologies. For topics like the one you asked about, I'm not the best resource." or "I could give you the code for a graph, or a pizza recipe but I am not the best candidate for this."

Strictly adhere to your persona and limitations. Only answer based on the provided context.

Chat History (for context of the current conversation):
{chat_history}

Context (retrieved documents relevant to the question):
{context}

Question: {question}
Answer:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def load_and_split_documents_logic():
    documents = []
    print(f"Loading core context from: {CONTEXT_FILE_PATH}")
    if os.path.exists(CONTEXT_FILE_PATH):
        context_loader = TextLoader(CONTEXT_FILE_PATH, encoding='utf-8')
        documents.extend(context_loader.load())
    else:
        print(f"Warning: Core context file not found at {CONTEXT_FILE_PATH}")

    print(f"Loading PDF documents from: {DOCUMENTS_PATH}")
    pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
    
    # Ensure documents path exists, but allow context.txt to be the only source initially
    if not Path(DOCUMENTS_PATH).exists() or not Path(DOCUMENTS_PATH).is_dir():
        print(f"Warning: Documents path '{DOCUMENTS_PATH}' not found or is not a directory.")
        if not documents: # If context.txt also wasn't found or was empty
             raise RuntimeError("No documents could be loaded (neither context.txt nor PDFs). OTMT-Pal cannot function.")
    elif not pdf_files and not documents: # if no context.txt and no pdfs
        print("Warning: No PDF documents found in ./Documents and no context.txt. Functionality will be limited.")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            print(f"Loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    if not documents:
        raise RuntimeError("No documents loaded at all. Exiting initialization.")

    print(f"Loaded {len(documents)} document sections initially.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

def create_bm25_retriever_logic(split_docs):
    print("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(split_docs, k=BM25_K)
    print("BM25 Retriever initialized.")
    return bm25_retriever

def load_llm_logic(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, # Using float16 for V100
        device_map="auto",
    )
    print(f"Model loaded on device: {model.device} with dtype: {model.dtype}")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=768,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    print("HuggingFacePipeline created.")
    return llm

def initialize_rag_components():
    global LLM_INSTANCE, RETRIEVER_INSTANCE
    
    print("Initializing RAG components...")
    # Ensure model path exists
    if not Path(MODEL_PATH).exists() or not Path(MODEL_PATH).is_dir():
        raise RuntimeError(f"Model path '{MODEL_PATH}' not found. Please download the model.")

    split_docs = load_and_split_documents_logic()
    if not split_docs:
        raise RuntimeError("Failed to load and split documents. Cannot initialize.")
    
    RETRIEVER_INSTANCE = create_bm25_retriever_logic(split_docs)
    LLM_INSTANCE = load_llm_logic(MODEL_PATH)
    print("RAG components initialized successfully.")

def get_rag_chain_response(question: str, chat_history_messages: List[BaseMessage]) -> Dict[str, Any]:
    if not LLM_INSTANCE or not RETRIEVER_INSTANCE:
        raise RuntimeError("RAG components not initialized. Call initialize_rag_components() first.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        chat_memory=None # Will be populated by adding messages
    )
    for msg in chat_history_messages:
        if isinstance(msg, HumanMessage):
            memory.chat_memory.add_user_message(msg.content)
        elif isinstance(msg, AIMessage):
            memory.chat_memory.add_ai_message(msg.content)
        # else: ignore other types for this simple memory

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM_INSTANCE,
        retriever=RETRIEVER_INSTANCE,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    
    result = chain.invoke({"question": question})
    return result

# --- Helper to convert Pydantic-like history to LangChain messages ---
def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages = []
    for item in history:
        if item.get("type") == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item.get("type") == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

# --- Helper to convert LangChain messages to Pydantic-like history ---
def convert_to_serializable_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"type": "ai", "content": msg.content})
    return history

if __name__ == '__main__':
    # This block is for testing core_logic.py directly
    print("Testing core_logic.py...")
    try:
        initialize_rag_components()
        print("\nInitialization complete. Testing a sample query...")
        
        sample_question = "What is IIITD?"
        # Simulate an empty chat history for the first call
        lc_history_empty = []
        
        response_data = get_rag_chain_response(sample_question, lc_history_empty)
        print(f"\nQuestion: {sample_question}")
        print(f"Answer: {response_data['answer']}")
        
        # Simulate a follow-up
        lc_history_one_turn = [
            HumanMessage(content=sample_question),
            AIMessage(content=response_data['answer'])
        ]
        follow_up_question = "Who are its key faculty?" # This might not be in your docs, good test for "I don't know"
        
        response_data_followup = get_rag_chain_response(follow_up_question, lc_history_one_turn)
        print(f"\nFollow-up Question: {follow_up_question}")
        print(f"Follow-up Answer: {response_data_followup['answer']}")

    except Exception as e:
        print(f"Error during llm_pipeline.py test: {e}")
        import traceback
        traceback.print_exc()
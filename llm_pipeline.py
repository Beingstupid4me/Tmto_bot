import os
import glob
import torch
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure basic logging (can be set to WARNING for less output later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as transformers_pipeline

# --- Configuration ---
# CRITICAL: Ensure this path points to a CHAT or INSTRUCT model, not a BASE model.
# For example: "Qwen/Qwen1.5-7B-Chat" or your local equivalent path.
# If MODEL_PATH points to a BASE model, prompt leaking is expected.
MODEL_PATH = "../Qwen3-8B-Base" # Replace with your CHAT/INSTRUCT model path if this is a base model
DOCUMENTS_PATH = "./Documents"
CONTEXT_FILE_PATH = "./context.txt"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BM25_K = 5 # Number of documents for BM25

# --- Global Instances (initialized once) ---
LLM_INSTANCE = None
RETRIEVER_INSTANCE = None

# --- Prompt Template ---
# Added more emphasis on staying in scope.
_template = """You are "OTMT-Pal", an AI assistant for the Office of Technology Management and Transfer (OTMT) at IIIT-Delhi.
Your developers are Amartya Singh (amartya22062@iiitd.ac.in) and Anish Dev (anish22075@iiitd.ac.in).
You are built using Langchain and a Qwen model. Your SOLE PURPOSE is to provide information about IIIT-Delhi, OTMT, its processes, Technology Readiness Levels (TRL), and technologies developed at the institute, based ONLY on the documents provided.

You MUST use the provided "Context" (retrieved documents) to answer the "Question".
STRICTLY ADHERE to the following:
1.  If the Context does not contain the answer, state "I do not have enough information to answer that specific question from the provided documents."
2.  If the question is outside your designated scope (IIIT-Delhi, OTMT, TRL, IIITD technologies), you MUST politely decline. For example, say: "My purpose is to assist with information about IIIT-Delhi's OTMT and its technologies. I cannot help with topics like [topic of unrelated question]."
3.  DO NOT answer general knowledge questions, give advice on unrelated topics (e.g., how to sing, pizza recipes, coding unrelated to IIITD tech), or provide information not found in the Context.
4.  DO NOT make up information.
5.  Begin your answer directly without repeating or rephrasing my instructions or persona.

Chat History (for context of the current conversation):
{chat_history}

Context (retrieved documents relevant to the question):
{context}

Question: {question}
Answer: 
/think
""" # Ensure no extra newlines or spaces after "Answer:"
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def load_and_split_documents_logic():
    # ... (no changes here, keep existing logging or reduce later) ...
    documents = []
    logging.info(f"Loading core context from: {CONTEXT_FILE_PATH}")
    if os.path.exists(CONTEXT_FILE_PATH):
        context_loader = TextLoader(CONTEXT_FILE_PATH, encoding='utf-8')
        documents.extend(context_loader.load())
    else:
        logging.warning(f"Core context file not found at {CONTEXT_FILE_PATH}")

    logging.info(f"Loading PDF documents from: {DOCUMENTS_PATH}")
    pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
    
    if not Path(DOCUMENTS_PATH).exists() or not Path(DOCUMENTS_PATH).is_dir():
        logging.warning(f"Documents path '{DOCUMENTS_PATH}' not found or is not a directory.")
        if not documents:
             raise RuntimeError("No documents could be loaded (neither context.txt nor PDFs). OTMT-Pal cannot function.")
    elif not pdf_files and not documents:
        logging.warning("No PDF documents found in ./Documents and no context.txt. Functionality will be limited.")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            logging.info(f"Loaded {pdf_file}")
        except Exception as e:
            logging.error(f"Error loading {pdf_file}: {e}")

    if not documents:
        raise RuntimeError("No documents loaded at all. Exiting initialization.")

    logging.info(f"Loaded {len(documents)} document sections initially.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks.")
    return split_docs

def create_bm25_retriever_logic(split_docs):
    # ... (no changes here) ...
    logging.info("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(split_docs, k=BM25_K)
    logging.info("BM25 Retriever initialized.")
    return bm25_retriever


def load_llm_logic(model_path_arg):
    logging.info(f"Loading model from: {model_path_arg}")
    tokenizer = AutoTokenizer.from_pretrained(model_path_arg)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_arg,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logging.info(f"Model loaded on device: {model.device} with dtype: {model.dtype} (8-bit quantization enabled)")

    hf_transformers_pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32768 # Should be enough for concise answers
    )
    llm = HuggingFacePipeline(pipeline=hf_transformers_pipeline)
    logging.info("HuggingFacePipeline (from langchain-huggingface) created.")
    return llm

def initialize_rag_components():
    # ... (no changes here) ...
    global LLM_INSTANCE, RETRIEVER_INSTANCE
    
    logging.info("Initializing RAG components...")
    if not Path(MODEL_PATH).exists() or not Path(MODEL_PATH).is_dir():
        # More informative error if it's the base model issue
        if "Base" in MODEL_PATH:
             logging.error(f"Model path '{MODEL_PATH}' seems to point to a BASE model. For chat/RAG, please use a CHAT or INSTRUCT fine-tuned model.")
        raise RuntimeError(f"Model path '{MODEL_PATH}' not found or is not a directory.")

    split_docs = load_and_split_documents_logic()
    if not split_docs:
        raise RuntimeError("Failed to load and split documents. Cannot initialize.")
    
    RETRIEVER_INSTANCE = create_bm25_retriever_logic(split_docs)
    LLM_INSTANCE = load_llm_logic(MODEL_PATH)
    logging.info("RAG components initialized successfully.")


def get_rag_chain_response(question: str, chat_history_messages: List[BaseMessage]) -> Dict[str, Any]:
    if not LLM_INSTANCE or not RETRIEVER_INSTANCE:
        logging.error("RAG components not initialized attempt to call get_rag_chain_response.")
        raise RuntimeError("RAG components not initialized. Call initialize_rag_components() first.")

    logging.info(f"Getting RAG chain response for question: '{question}'")
    # logging.info(f"Number of messages in provided history: {len(chat_history_messages)}") # Can be noisy

    current_chat_history_obj = ChatMessageHistory()
    for msg in chat_history_messages:
        if isinstance(msg, HumanMessage):
            current_chat_history_obj.add_user_message(msg.content)
        elif isinstance(msg, AIMessage):
            current_chat_history_obj.add_ai_message(msg.content)
    # logging.info("ChatMessageHistory object created and populated.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        chat_memory=current_chat_history_obj
    )
    # logging.info(f"ConversationBufferMemory initialized. Memory buffer: {memory.load_memory_variables({})}") # Can be noisy

    # logging.info("Creating ConversationalRetrievalChain...") # Can be noisy
    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM_INSTANCE,
        retriever=RETRIEVER_INSTANCE,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT},
        return_source_documents=True,
        verbose=False # <<< SET CHAIN TO VERBOSE=FALSE
    )
    # logging.info("ConversationalRetrievalChain created.")
    
    logging.info(f"Invoking chain with question (first 50 chars): '{question[:50]}...'")
    try:
        result = chain.invoke({"question": question})
        logging.info("Chain invocation complete.")
        return result
    except Exception as e:
        logging.error(f"Exception during chain invocation: {e}", exc_info=True)
        raise

# --- Helper functions (no changes) ---
def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages = []
    for item in history:
        if item.get("type") == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item.get("type") == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

def convert_to_serializable_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"type": "ai", "content": msg.content})
    return history

if __name__ == '__main__':
    # ... (test script can remain for standalone testing, or be commented out) ...
    logging.info("Starting llm_pipeline.py test script...")
    try:
        initialize_rag_components()
        logging.info("Initialization complete. Testing a simple sample query...")
        
        sample_question = "Hello, who are you?"
        lc_history_empty = []
        
        logging.info(f"Test 1: Sending question '{sample_question}' with empty history.")
        response_data = get_rag_chain_response(sample_question, lc_history_empty)
        
        logging.info(f"\nQuestion: {sample_question}")
        logging.info(f"Answer: {response_data['answer']}")
        
        logging.info("Test 2: Simulating a follow-up question.")
        lc_history_one_turn = [
            HumanMessage(content=sample_question),
            AIMessage(content=response_data['answer'])
        ]
        follow_up_question = "How do I file a patent at IIITD?" # More relevant question
        
        logging.info(f"Sending follow-up question '{follow_up_question}' with one turn history.")
        response_data_followup = get_rag_chain_response(follow_up_question, lc_history_one_turn)
        
        logging.info(f"\nFollow-up Question: {follow_up_question}")
        logging.info(f"Follow-up Answer: {response_data_followup['answer']}")

    except Exception as e:
        logging.error(f"Error during llm_pipeline.py test: {e}", exc_info=True)
    logging.info("llm_pipeline.py test script finished.")

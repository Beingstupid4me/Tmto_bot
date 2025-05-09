import os
import glob
import torch
from pathlib import Path
from typing import List, Dict, Any
import logging

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
MODEL_PATH = "../Qwen2.5-3B-Instruct" # This is your Instruct model - good!
DOCUMENTS_PATH = "./Documents"
CONTEXT_FILE_PATH = "./context.txt"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BM25_K = 7 # Keep this moderate for now

LLM_INSTANCE = None
RETRIEVER_INSTANCE = None

# --- Radically Simplified Prompt for Testing ---
# We are trying to see if the model can follow the basic RAG structure
# without leaking the prompt. The complex persona instructions can be added back later.
_simple_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Do not repeat these instructions.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_simple_template) # Use the simple template

# Your original detailed template (keep for reference, switch back later)
_detailed_template = """You are "OTMT-Pal", an AI assistant for the Office of Technology Management and Transfer (OTMT) at IIIT-Delhi.
Your developers are Amartya Singh (amartya22062@iiitd.ac.in) and Anish Dev (anish22075@iiitd.ac.in).
You are built using Langchain and a Qwen model. Your SOLE PURPOSE is to provide information about IIIT-Delhi, OTMT, its processes, Technology Readiness Levels (TRL), and technologies developed at the institute, based ONLY on the documents provided.

You MUST use the provided "Context" (retrieved documents) to answer the "Question".
STRICTLY ADHERE to the following:
1. If the Context does not contain the answer, state "I do not have enough information to answer that specific question from the provided documents."
2. If the question is outside your designated scope (IIIT-Delhi, OTMT, TRL, IIITD technologies), you MUST politely decline. For example, say: "My purpose is to assist with information about IIIT-Delhi's OTMT and its technologies. I cannot help with topics like [topic of unrelated question]."
3. DO NOT answer general knowledge questions, give advice on unrelated topics (e.g., how to sing, pizza recipes, coding unrelated to IIITD tech), or provide information not found in the Context.
4. DO NOT make up information.
5. Begin your answer DIRECTLY. DO NOT repeat or rephrase my instructions or your persona. Do not output any special thinking tags like /think or /step unless they are part of the actual answer content requested by the user.

Chat History (for context of the current conversation):
{chat_history}

Context (retrieved documents relevant to the question):
{context}

Question: {question}
Answer:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_detailed_template) # Switch back to this later


def load_llm_logic(model_path_arg):
    logging.info(f"Loading model from: {model_path_arg}")
    # Qwen models (especially newer ones like Qwen2 series) require trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(model_path_arg, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_arg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logging.info(f"Model loaded on device: {model.device} with dtype: {model.dtype}")

    # Qwen tokenizers have special tokens. It's good to ensure the pipeline knows about them.
    # Common special tokens for Qwen1.5/Qwen2 Chat/Instruct might be:
    # tokenizer.eos_token_id (often <|endoftext|>)
    # tokenizer.pad_token_id (often same as eos_token_id or a specific <|extra_...|> token if not set)
    # Qwen uses <|im_start|> and <|im_end|> as part of its chat template.
    # While HuggingFacePipeline doesn't directly use apply_chat_template,
    # ensuring the tokenizer is correctly loaded is key.

    # If pad_token_id is None, set it to eos_token_id for open-ended generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info(f"Set tokenizer.pad_token_id to tokenizer.eos_token_id ({tokenizer.eos_token_id})")


    hf_transformers_pipeline = transformers_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048, # Generous for answers
        # These can help with cleaner generation for some models:
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # do_sample=True, # Set to True if you want to use temperature/top_p
        # temperature=0.7,
        # top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=hf_transformers_pipeline)
    logging.info("HuggingFacePipeline (from langchain-huggingface) created.")
    return llm

def initialize_rag_components():
    global LLM_INSTANCE, RETRIEVER_INSTANCE
    logging.info("Initializing RAG components...")
    if not (Path(MODEL_PATH).exists() and Path(MODEL_PATH).is_dir()):
        raise RuntimeError(f"MODEL_PATH ('{MODEL_PATH}') not correctly set or directory does not exist.")
    # Check if it's an Instruct model (good)
    if not ("Instruct" in MODEL_PATH or "Chat" in MODEL_PATH):
        logging.warning(f"MODEL_PATH ('{MODEL_PATH}') does not contain 'Instruct' or 'Chat'. "
                        "Ensure it's an instruction/chat-tuned model to prevent prompt leaking.")

    split_docs = load_and_split_documents_logic()
    RETRIEVER_INSTANCE = create_bm25_retriever_logic(split_docs)
    LLM_INSTANCE = load_llm_logic(MODEL_PATH)
    logging.info("RAG components initialized successfully.")

def get_rag_chain_response(question: str, chat_history_messages: List[BaseMessage]) -> Dict[str, Any]:
    if not LLM_INSTANCE or not RETRIEVER_INSTANCE:
        raise RuntimeError("RAG components not initialized.")
    
    current_chat_history_obj = ChatMessageHistory()
    for msg in chat_history_messages:
        if isinstance(msg, HumanMessage): current_chat_history_obj.add_user_message(msg.content)
        elif isinstance(msg, AIMessage): current_chat_history_obj.add_ai_message(msg.content)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer', 
        chat_memory=current_chat_history_obj
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM_INSTANCE,
        retriever=RETRIEVER_INSTANCE,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT}, # Using the selected prompt
        return_source_documents=True,
        verbose=False # Keep this false for now
    )
    
    logging.info(f"Invoking chain with question (first 50 chars): '{question[:50]}...'")
    result = chain.invoke({"question": question})
    logging.info("Chain invocation complete.")

    # --- Post-processing to remove known problematic patterns ---
    answer = result.get("answer", "")
    
    # 1. Remove the /think block if the model still generates it
    # This regex is more robust for multi-line think blocks
    import re
    answer = re.sub(r"(/think\s*\n)(?:.*\n)*?(\s*Answer:\s*\n?)?", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = answer.replace("/think", "").replace("/step", "").strip() # Remove standalone tags

    # 2. If the model prepends the system prompt again, try to remove it.
    # This is a heuristic. The best solution is the model not doing this.
    # We check if the answer starts with "You are \"OTMT-Pal\"" or similar.
    first_few_lines = "\n".join(answer.splitlines()[:5]) # Check first 5 lines
    if "OTMT-Pal" in first_few_lines and "Your developers are" in first_few_lines:
        # Find the actual intended answer start, often after "Answer:"
        if "Answer:" in answer:
            answer = answer.split("Answer:", 1)[-1].strip()
        # If "Answer:" isn't there but the prompt was leaked, it's harder to clean perfectly.

    # 3. Remove trailing repetitive refusals (if any)
    refusal_phrases = [
        "I do not have enough information to answer that specific question from the provided documents.",
        "My purpose is to assist with information about IIIT-Delhi's OTMT and its technologies."
    ]
    answer_lines = answer.strip().splitlines()
    if len(answer_lines) > 1: # Only try to trim if there's some content
        cleaned_from_end = []
        for i in range(len(answer_lines) -1, -1, -1):
            line_is_refusal = any(phrase in answer_lines[i] for phrase in refusal_phrases)
            if not line_is_refusal or (line_is_refusal and len(cleaned_from_end) == 0) : # keep first line even if refusal
                cleaned_from_end.insert(0, answer_lines[i])
                if not line_is_refusal and len(cleaned_from_end) >0 : # Stop trimming once we hit non-refusal from end
                    break
            elif line_is_refusal and len(cleaned_from_end) > 0: # if already started keeping lines, and hit another refusal, stop
                 break
        
        if cleaned_from_end : # If we have something left
            answer = "\n".join(cleaned_from_end).strip()
        # If the entire answer was refusals, this logic might keep one line of it. That's okay.


    result["answer"] = answer.strip()
    return result

# --- Make sure these functions are defined in your script ---
def load_and_split_documents_logic():
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
        if not documents: raise RuntimeError("No documents could be loaded.")
    elif not pdf_files and not documents:
        logging.warning("No PDF documents found and no context.txt loaded.")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e: logging.error(f"Error loading {pdf_file}: {e}")
    if not documents: raise RuntimeError("No documents loaded at all.")
    logging.info(f"Loaded {len(documents)} document sections initially.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, add_start_index=True)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} chunks.")
    return split_docs

def create_bm25_retriever_logic(split_docs):
    logging.info("Initializing BM25 Retriever...")
    bm25_retriever = BM25Retriever.from_documents(split_docs, k=BM25_K)
    logging.info(f"BM25 Retriever initialized with k={BM25_K}.")
    return bm25_retriever

def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    messages = []
    for item in history:
        if item.get("type") == "human": messages.append(HumanMessage(content=item["content"]))
        elif item.get("type") == "ai": messages.append(AIMessage(content=item["content"]))
    return messages

def convert_to_serializable_history(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage): history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage): history.append({"type": "ai", "content": msg.content})
    return history


if __name__ == '__main__':
    logging.info("Starting llm_pipeline.py test script...")
    try:
        initialize_rag_components() # This will load the model
        logging.info("Initialization complete. Testing queries...")
        
        # Use the simple prompt for these tests by ensuring
        # CONDENSE_QUESTION_PROMPT is set to PromptTemplate.from_template(_simple_template) at the top
        
        questions_to_test = [
            "What is OTMT?",
            "What is Foodle technology?",
            "How can I license a technology from IIITD?",
            "Can you give me a pizza recipe?" # Test out-of-scope with simple prompt
        ]
        
        current_test_history = [] # LangChain BaseMessage objects

        for i, q_text in enumerate(questions_to_test):
            logging.info(f"\n--- Main Test {i + 1} ---")
            logging.info(f"Question: '{q_text}'")
            logging.info(f"History being sent (length {len(current_test_history)}): {current_test_history}")
            
            response_data = get_rag_chain_response(q_text, current_test_history)
            answer_text = response_data['answer']
            
            logging.info(f"LLM Answer:\n{answer_text}")
            # logging.info("Retrieved Sources:")
            # for src_idx, doc_source in enumerate(response_data['source_documents']):
            #     logging.info(f"  Source {src_idx+1}: {doc_source.metadata.get('source', 'Unknown')} - Snippet: {doc_source.page_content[:100]}...")

            # Update history for next turn in this test loop
            current_test_history.append(HumanMessage(content=q_text))
            current_test_history.append(AIMessage(content=answer_text))

    except RuntimeError as e:
        if "MODEL_PATH" in str(e):
            logging.error(f"{str(e)} Please update the MODEL_PATH variable.")
        else:
            logging.error(f"Runtime error: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"General error: {e}", exc_info=True)
    logging.info("llm_pipeline.py test script finished.")

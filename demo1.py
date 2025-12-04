import sys
import json
import os
import uuid
import numpy as np

# --- LangChain Imports ---
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# --- LangGraph Imports ---
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ==================================================================
# 1. DEFINE GRAPH STATE
# ==================================================================

class GraphState(TypedDict):
    """
    Represents the state of our RAG workflow.
    """
    messages: Annotated[List[BaseMessage], add_messages] 
    question: str  # The active query (may be rewritten)
    original_question: str # Keep track of what user actually typed
    query_embedding: Optional[List[float]]
    documents: Optional[List[Document]] 
    answer: Optional[str]
    cache_hit: bool
    user_id: str   # Tracks specific user for isolation

# ==================================================================
# 2. RAG PIPELINE CLASS
# ==================================================================

class RAGPipeline:
    
    def __init__(self):
        print("[Startup]: RUNNING STARTUP (OLLAMA + REWRITE + MULTI-USER)")
        
        # --- Configuration ---
        self.OLLAMA_BASE_URL = "http://localhost:11434"
        self.LLM_MODEL_NAME = "llama3.1:8b" 
        self.EMBED_MODEL_NAME = "nomic-embed-text:latest" 
        
        self.DATA_DIR = "data"
        self.CHROMA_DIR = "./chroma_db"
        
        # Multi-User Cache Directory
        self.CACHE_DIR = "user_caches"
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
        
        # RRF Settings
        self.RETRIEVER_TOP_K = 10 
        self.FINAL_TOP_K = 8      
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # --- Initialization ---
        self.embed_model = self.setup_embeddings()
        self.llm = self.setup_llm()
        
        # --- Load Data ---
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
            
        self.documents = self.load_documents(self.DATA_DIR) or []
        if self.documents:
            self.chunks = self.chunk_documents(self.documents)
            print(f"[Chunks]: Total documents loaded: {len(self.documents)}. Total chunks created: {len(self.chunks)}.")
        else:
            print(f"[Chunks]: Warning: No documents found in 'data' folder. Please add .txt files there.")
            self.chunks = []
        
        # --- Setup Retrievers ---
        if self.chunks:
            print("[Retreiver]: Setting up retrievers (BM25 + Chroma)...")
            self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
            self.bm25_retriever.k = self.RETRIEVER_TOP_K
            self.vector_store = self.create_vector_store(self.chunks)
        else:
            self.bm25_retriever = None
            self.vector_store = None
        
        # --- Memory & Graph ---
        self.memory = MemorySaver() 
        self.app = self.create_langgraph_workflow()
        print("[Pipeline]: STARTUP COMPLETE. Pipeline is ready.")

    # --- Helper Methods ---
    
    def setup_embeddings(self):
        return OllamaEmbeddings(
            model=self.EMBED_MODEL_NAME,
            base_url=self.OLLAMA_BASE_URL
        )

    def setup_llm(self):
        return ChatOllama(
            model=self.LLM_MODEL_NAME,
            base_url=self.OLLAMA_BASE_URL,
            temperature=0
        )

    def load_documents(self, folder_path):
        try:
            docs = DirectoryLoader(folder_path, glob="**/*.txt").load()
            print("[Data]: Data loaded from files")
            return docs
        except Exception as e: 
            print(f"[Data]: Error while loading files {e}")
            return None

    def chunk_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, chunks):
        vector_store = Chroma(
            persist_directory=self.CHROMA_DIR, 
            embedding_function=self.embed_model, 
            collection_name="rag_collection"
        )
        if vector_store._collection.count() == 0: 
            print(f"[Indexing]: creating index of {len(chunks)} chunks into ChromaDB...")
            vector_store.add_documents(chunks)
        else:
            print(f"[Indexing]: ChromaDB index already exists with {vector_store._collection.count()} vectors.")
        return vector_store

    def get_user_cache_path(self, user_id):
        """Helper to get the specific filename for a user."""
        safe_id = "".join([c for c in user_id if c.isalnum() or c in ('-', '_')]) 
        return os.path.join(self.CACHE_DIR, f"{safe_id}_cache.json")

    # ==============================================================
    # 3. RECIPROCAL RANK FUSION (RRF)
    # ==============================================================
    
    def reciprocal_rank_fusion(self, results: list[list], k=60):
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc.page_content
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = {"doc": doc, "score": 0}
                fused_scores[doc_str]["score"] += 1.0 / (k + rank)
        
        reranked_results = sorted(
            fused_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        return [item["doc"] for item in reranked_results]

    # ==============================================================
    # 4. LANGGRAPH NODES
    # ==============================================================

    def check_cache_node(self, state: GraphState):
        """NODE 1: CACHE LOOKUP"""
        
        user_id = state["user_id"]
        messages = state["messages"]
        last_message = messages[-1]
        
        # We capture the raw question here
        original_question = last_message.content
        
        # 1. Load specific user cache
        user_cache_path = self.get_user_cache_path(user_id)
        user_cache_data = {}

        if os.path.exists(user_cache_path):
            try:
                with open(user_cache_path, 'r') as f:
                    user_cache_data = json.load(f)
            except json.JSONDecodeError: pass

        # 2. Check Cache (Exact match on original question)
        if original_question in user_cache_data:
             cached_answer = user_cache_data[original_question]["answer"]
             print(f"[Cache]: CACHE HIT for User '{user_id}' ---")
             return {
                 "answer": cached_answer, 
                 "cache_hit": True, 
                 "messages": [AIMessage(content=cached_answer)],
                 "question": original_question,
                 "original_question": original_question
             }

        # Cache Miss
        return {
            "cache_hit": False, 
            "question": original_question, 
            "original_question": original_question
        }

    def rewrite_query_node(self, state: GraphState):
        """NODE 2: REWRITE QUERY (Improve context)"""
        
        question = state["question"]
        messages = state["messages"]
        
        # If no history (len <= 1), user just started. No need to rewrite.
        if len(messages) <= 1:
            print("[Rewrite]: No history, skipping rewrite.")
            return {"question": question}

        # Prompt for Rewriting
        system_prompt = """You are an expert at refining search queries.
        The user is asking a follow-up question. 
        Rewrite the "Latest Question" into a standalone, specific question using the "Chat History".
        
        RULES:
        1. Do NOT answer the question.
        2. Return ONLY the rewritten question string.
        3. If the question is already clear, return it as is.
        """
        
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]])
        human_prompt = f"""
        Chat History:
        {history_str}
        
        Latest Question: 
        {question}
        
        Standalone Question:"""
        
        msg = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        # Call LLM
        response = self.llm.invoke(msg)
        rewritten_query = response.content.strip()
        
        print(f"[Rewrite]: Original: {question} \nRewritten: {rewritten_query}")
        
        # We update 'question' so the next node (retrieval) uses the better version
        return {"question": rewritten_query}

    def context_generation_node(self, state: GraphState):
        """NODE 3: RETRIEVAL"""
        question = state["question"] # This is now the REWRITTEN question
        
        # Embed the rewritten question
        query_embedding = self.embed_model.embed_query(question)
        
        if not self.vector_store:
            return {"documents": [], "query_embedding": query_embedding}

        # 1. Run Retrievers
        bm25_docs = self.bm25_retriever.invoke(question)
        chroma_docs = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=self.RETRIEVER_TOP_K
        )
        
        # 2. RRF
        fused_docs = self.reciprocal_rank_fusion([bm25_docs, chroma_docs])
        final_docs = fused_docs[:self.FINAL_TOP_K]
        print(f"[Context]: {final_docs}")
        
        return {"documents": final_docs, "query_embedding": query_embedding}

    def response_generation_node(self, state: GraphState):
        """NODE 4: GENERATION"""
        documents = state["documents"]
        
        context_str = "\n\n".join([f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(documents)])
        
        system_prompt_content = f"""
        You are a helpful assistant. Use the retrieved context to answer the user's question.
        
        Context:
        {context_str}
        """
        
        messages = [SystemMessage(content=system_prompt_content)] + state["messages"]
        
        response_content = ""
        for chunk in self.llm.stream(messages):
            response_content += chunk.content
        
        return {
            "answer": response_content,
            "messages": [AIMessage(content=response_content)]
        }

    def cache_update_node(self, state: GraphState):
        """NODE 5: CACHE UPDATE"""
        user_id = state["user_id"]
        original_q = state["original_question"] # Key by what the user TYPED, not the rewrite
        
        if state["answer"]:
            user_cache_path = self.get_user_cache_path(user_id)
            current_cache = {}
            
            if os.path.exists(user_cache_path):
                try:
                    with open(user_cache_path, 'r') as f:
                        current_cache = json.load(f)
                except: pass

            current_cache[original_q] = {
                "answer": state["answer"],
                # We save the embedding of the REWRITTEN query, as it's higher quality
                "embedding": state["query_embedding"] 
            }
            
            try:
                with open(user_cache_path, 'w') as f:
                    json.dump(current_cache, f, indent=2)
                    print(f"\n[Cache]: Cache saved successfully for User '{user_id}'.")
            except Exception as e:
                print(f"\n[Cache]: Cache save failed: {e}")
        return {}

    # --- Graph Definition ---
    def create_langgraph_workflow(self):
        workflow = StateGraph(GraphState)
        
        # Add Nodes
        workflow.add_node("check_cache", self.check_cache_node)
        workflow.add_node("rewrite_query", self.rewrite_query_node)
        workflow.add_node("context_generation", self.context_generation_node)
        workflow.add_node("response_generation", self.response_generation_node)
        workflow.add_node("cache_update", self.cache_update_node)

        # Logic
        workflow.add_edge(START, "check_cache")
        
        def should_run_rag(state):
            return "run_rag" if not state["cache_hit"] else "end"
            
        workflow.add_conditional_edges(
            "check_cache",
            should_run_rag,
            {
                "run_rag": "rewrite_query", 
                "end": END
            }
        )
        
        workflow.add_edge("rewrite_query", "context_generation")
        workflow.add_edge("context_generation", "response_generation")
        workflow.add_edge("response_generation", "cache_update")
        workflow.add_edge("cache_update", END)

        return workflow.compile(checkpointer=self.memory)

    # --- Run Method ---
    def run_chat(self):
        
        current_user_id = input("Enter User ID (e.g., 'alice', 'bob'): ").strip()
        if not current_user_id: current_user_id = "default_user"
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"Logged in as: {current_user_id}")
        print(f"Session ID: {thread_id}")
        print("Type 'quit' to exit.")
        
        while True:
            user_input = input(f"\n{current_user_id}: ")
            if user_input.lower() in ["quit", "exit"]: 
                break
            if not user_input.strip(): 
                continue
            
            # Pass inputs to graph
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "user_id": current_user_id
            }
            
            final_answer = ""
            try:
                for event in self.app.stream(inputs, config, stream_mode="values"):
                    messages = event.get("messages", [])
                    if messages and isinstance(messages[-1], AIMessage):
                        current_content = messages[-1].content
                        if current_content != final_answer:
                            print(f"\rAI: {current_content}", end="", flush=True)
                            final_answer = current_content
                            
            except Exception as e:
                print(f"Error during execution: {e}") 
if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()
        pipeline.run_chat()
            
    except Exception as e:
        print(f"\nFatal Error: {e}")

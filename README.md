# RAG-chatbot
Rag chatbot for college information retrieval 
Tech Stack: Python, Hugging Face Transformers, LangChain, FAISS, Sentence Transformers, Mistral-7B
Instruct 
 Developed a Retrieval-Augmented Generation (RAG) chatbot to assist students with college-related queries 
including admissions, courses, departments, and campus information. 
 Implemented semantic search using FAISS and sentence-transformers/all-mpnet-base-v2 for accurate 
document retrieval. 
 Fine-tuned and deployed Mistral-7B-Instruct-v0.1-AWQ as the LLM to generate context-aware answers from 
retrieved chunks. 
 Leveraged LangChain for chaining document loaders, embedding models, vector stores, and LLM pipelines. 
 Used RecursiveCharacterTextSplitter to chunk large documents while preserving context for improved 
embedding and retrieval. 
 Built a retrieval QA pipeline using LangChain's RetrievalQA module to integrate the LLM with the vector store. 
 Ensured performance and compatibility using transformers, AutoTokenizer, and AutoModelForCausalLM 
with optimized quantized models. 
 Employed TextLoader for ingesting plain text documents into the knowledge base with modular pre-processing. 
 Designed a scalable architecture for potential deployment via FastAPI or Streamlit for frontend interaction. 
 Focused on cost-efficiency by running the inference pipeline on CPU after GPU-based model preparation. 

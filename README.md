College Information RAG Chatbot
A Retrieval-Augmented Generation (RAG) based chatbot designed to assist students with college-related queries, such as admissions, departments, courses, and campus life. Built using powerful open-source tools like Mistral-7B-Instruct, FAISS, LangChain, and Sentence Transformers.

Tech Stack
Programming Language: Python

Large Language Model: Mistral-7B-Instruct-v0.1-AWQ (quantized)

Semantic Search: FAISS + sentence-transformers/all-mpnet-base-v2

Frameworks & Libraries:

Hugging Face Transformers

LangChain

FAISS

Sentence Transformers

Document Processing: RecursiveCharacterTextSplitter, TextLoader

Deployment Ready With: FastAPI / Streamlit

Key Features
💬 Context-Aware Answering: Uses a fine-tuned Mistral-7B-Instruct LLM to answer student queries.

🔍 Semantic Search: Accurately retrieves relevant documents using FAISS and dense embeddings.

🧩 Chunking with Context: Utilizes RecursiveCharacterTextSplitter to retain contextual integrity during document chunking.

🔄 Modular RAG Pipeline: Integrates embedding, vector search, and LLM via LangChain’s RetrievalQA.

⚡ Efficient Inference: Optimized quantized models ensure cost-efficient CPU-based deployment.

🏗️ Scalable Architecture: Backend-ready for deployment with FastAPI or Streamlit for interactive frontend experiences.

Project Structure
📁 rag-college-chatbot/
│
├── 📄 app.py                  # Backend FastAPI or Streamlit app
├── 📄 rag_pipeline.py         # LangChain-based RAG pipeline
├── 📄 vectorstore_builder.py  # FAISS vector store construction
├── 📄 llm_loader.py           # Load and configure Mistral-7B
├── 📄 requirements.txt        # Python dependencies
├── 📁 data/                   # Folder for raw and processed documents
├── 📁 models/                 # Folder to store quantized LLM models
└── 📄 README.md               # Project documentation

Installation
# Clone the repo
git clone https://github.com/your-username/rag-college-chatbot.git
cd rag-college-chatbot

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

Model Details
LLM: Mistral-7B-Instruct-v0.1-AWQ

Tokenizer: AutoTokenizer.from_pretrained()

Model Loader: AutoModelForCausalLM.from_pretrained() with quantized weights

Embeddings: all-mpnet-base-v2 from Sentence Transformers

Features in Detail
Feature	Description
Document Ingestion	TextLoader reads plain text files and prepares them for vectorization
Chunking	RecursiveCharacterTextSplitter maintains context across document chunks
Embeddings	Sentence-level embeddings for semantic similarity via all-mpnet-base-v2
Vector Store	Efficient FAISS index for fast nearest-neighbor search
LLM Inference	Mistral-7B-Instruct-v0.1-AWQ for generating responses
Retrieval QA Pipeline	Built using LangChain’s RetrievalQA module
Modular Design	Easy to extend or swap out components
Deployment Ready	Designed for integration with FastAPI or Streamlit

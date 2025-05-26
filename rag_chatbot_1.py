import os
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from llama_cpp import Llama

# Step 1: Load PDF
loader = PyPDFLoader(r"D:\Dhanwis\RAG_chatbot\college_data.pdf")
pages = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Step 3: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 4: Load a small LLM locally using Transformers
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)

# Wrap in LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Step 5: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 6: Ask questions
while True:
    query = input("Ask something (or 'exit'): ")
    if query.lower() == "exit":
        break
    print("Answer:", qa_chain.invoke(query))

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import List, Optional, Any
from llama_cpp import Llama
from pydantic import PrivateAttr


class LlamaCppLLM(LLM):
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 4

    _client: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads
        )

    @property
    def _llm_type(self) -> str:
        return "llama-cpp"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        output = self._client(prompt, stop=stop, **kwargs)
        return output["choices"][0]["text"].strip()

    @property
    def _identifying_params(self) -> dict:  # ✅ fixed this
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads
        }



# Step 1: Load PDF
loader = PyPDFLoader(r"D:\Dhanwis\RAG_chatbot\college_data.pdf")
pages = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Step 3: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 4: Load llama-cpp-python model
llm = LlamaCppLLM(
    model_path=r"D:\Dhanwis\RAG_chatbot\mistr_llm\tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    temperature=0.7,
    max_tokens=256,
    top_p=0.95,
    n_ctx=1024
)

# Step 5: Retrieval-based QA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 6: Interactive chat
while True:
    query = input("Ask something (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke({"query": query})
    print("Answer:", answer)

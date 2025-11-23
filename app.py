import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not directly used in the app logic, can be removed
import nltk
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- NLTK Data Download (Optional, uncomment if needed) ---
# NLTK data path. In a production environment, ensure this data is available.
# nltk.data.path.append("./nltk_data")
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

# --- Environment Variable Check ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# --- Data Loading and Processing ---
# Ensure the 'books' directory exists and contains the PDF files.
# You might need to manually extract "資料科學與回歸分析講義.7z" into a 'books' folder.
# Example: 7z x "資料科學與回歸分析講義.7z" -obooks
pdf_directory = "./books/資料科學與回歸分析講義"
if not os.path.exists(pdf_directory):
    print(f"Warning: Directory '{pdf_directory}' not found. Please ensure PDF files are extracted.")
    # Exit or handle gracefully if documents are essential
    # For now, we'll proceed, but document loading will fail.

loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

# --- Embeddings ---
# Using HuggingFace embeddings as per the last definition in the notebook
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- FAISS Vector Store and Retriever ---
FAISS_INDEX_PATH = "faiss_index"

if not os.path.exists(FAISS_INDEX_PATH):
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"FAISS vector store created and saved to '{FAISS_INDEX_PATH}'")
else:
    print(f"Loading existing FAISS vector store from '{FAISS_INDEX_PATH}'...")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS vector store loaded.")

retriever = vector_store.as_retriever()

# --- Classification Key Terms ---
classification_key_terms = [
    # --- 模型核心概念 ---
    "SVM (支持向量機)：核心概念是尋找一個最大邊界 (Maximum Margin) 的超平面 (Hyperplane) 來分隔不同類別的資料點。關鍵在於選擇合適的核函數 (Kernel Function)。",
    "隨機森林 (Random Forest)：一種集成學習 (Ensemble Learning) 方法，透過建立多棵決策樹 (Decision Trees) 並取多數決 (Voting) 來進行分類，有效減少過擬合 (Overfitting)。",
    "線性模型 (分類)：通常指的是邏輯迴歸 (Logistic Regression)，它使用 Sigmoid 函數將線性預測轉換為機率值，常用於二元分類，並定義決策邊界。",

    # --- 評估與優化 ---
    "評估指標：分類器的性能主要透過混淆矩陣 (Confusion Matrix)、準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall) 和 F1 Score 來評估。",
    "超參數調優 (Hyperparameter Tuning)：例如 SVM 的 C 參數和 Kernel 類型，或隨機森林的樹木數量，這些需要通過交叉驗證 (Cross-Validation) 進行優化。",

    # --- 回答指導原則 ---
    "當回答關於分類器的問題時，請先解釋該模型的**核心原理**與**關鍵參數**，並比較不同模型在處理**非線性資料**或**數據量大**時的優勢與劣勢。",
]
technical_guidance = "\n".join(classification_key_terms)

# --- RAG QA Chain ---
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Main Function ---
def answer_user_question(question: str) -> str:
    """
    Answers a user question using RAG with technical guidance.
    """
    # Use invoke instead of run for newer LangChain versions
    retriever_result = qa_chain.invoke({"query": question})['result']

    prompt = f"""
    你的任務是扮演一位專業的機器學習專家，根據以下技術指導原則和檢索到的文件內容來回答問題。

    請嚴格遵循以下技術指導原則：
    {technical_guidance}

    ---
    以下是我們從資料庫中檢索到的內容，這些內容來自書中的資料，並與使用者的問題相關：
    {retriever_result}
    ---

    請根據上述所有資料和指導原則，以專業、嚴謹的語氣，清晰地回應使用者的問題：
    「{question}」
    """

    final_response = llm.invoke([HumanMessage(content=prompt)])
    return final_response.content

# --- Example Usage ---
if __name__ == "__main__":
    user_question = "如果面對一個非線性資料集，SVM 中的核函數 (Kernel Function) 是什麼？它如何幫助解決這個問題？"
    print(f"User Question: {user_question}")
    response = answer_user_question(user_question)
    print(f'\n經過機器人得到的內容是 \n==================== \n{response}')

    # You can add more questions here for testing
    # user_question_2 = "隨機森林的優點是什麼？"
    # print(f"\nUser Question: {user_question_2}")
    # response_2 = answer_user_question(user_question_2)
    # print(f'\n經過機器人得到的內容是 \n==================== \n{response_2}')

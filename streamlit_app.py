import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Environment Variable Check ---
# Using st.secrets for Streamlit deployment, with a fallback for local development
try:
    # This will work when deployed on Streamlit Community Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # For local development, it will look for an environment variable
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your Streamlit secrets or as an environment variable.")
    st.stop()

# --- Load FAISS Vector Store and Retriever ---
@st.cache_resource
def load_retriever():
    FAISS_INDEX_PATH = "faiss_index"
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index directory '{FAISS_INDEX_PATH}' not found. Please ensure the index is in the correct location.")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        st.stop()

# --- LLM and QA Chain Setup ---
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

# --- Main Function to get answer ---
def get_answer(question: str, retriever, llm):
    """
    Answers a user question using RAG with technical guidance.
    """
    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Define technical guidance
    classification_key_terms = [
        "SVM (支持向量機)：核心概念是尋找一個最大邊界 (Maximum Margin) 的超平面 (Hyperplane) 來分隔不同類別的資料點。關鍵在於選擇合適的核函數 (Kernel Function)。",
        "隨機森林 (Random Forest)：一種集成學習 (Ensemble Learning) 方法，透過建立多棵決策樹 (Decision Trees) 並取多數決 (Voting) 來進行分類，有效減少過擬合 (Overfitting)。",
        "線性模型 (分類)：通常指的是邏輯迴歸 (Logistic Regression)，它使用 Sigmoid 函數將線性預測轉換為機率值，常用於二元分類，並定義決策邊界。",
        "評估指標：分類器的性能主要透過混淆矩陣 (Confusion Matrix)、準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall) 和 F1 Score 來評估。",
        "超參數調優 (Hyperparameter Tuning)：例如 SVM 的 C 參數和 Kernel 類型，或隨機森林的樹木數量，這些需要通過交叉驗證 (Cross-Validation) 進行優化。",
        "當回答關於分類器的問題時，請先解釋該模型的**核心原理**與**關鍵參數**，並比較不同模型在處理**非線性資料**或**數據量大**時的優勢與劣勢。",
    ]
    technical_guidance = "\n".join(classification_key_terms)

    # 3. Construct the prompt
    prompt = f"""
    你的任務是扮演一位專業的機器學習專家，根據以下技術指導原則和檢索到的文件內容來回答問題。

    請嚴格遵循以下技術指導原則：
    {technical_guidance}

    ---
    以下是我們從資料庫中檢索到的內容，這些內容來自書中的資料，並與使用者的問題相關：
    {context}
    ---

    請根據上述所有資料和指導原則，以專業、嚴謹的語氣，清晰地回應使用者的問題：
    「{question}」
    """

    # 4. Get the final response from the LLM
    final_response = llm.invoke([HumanMessage(content=prompt)])
    return final_response.content


# --- Streamlit App UI ---
# 設置頁面配置，加入圖示和更專業的標題
st.set_page_config(
    page_title="🤖 ML 智慧 RAG 分析專家",
    layout="wide" # 使用寬版佈局，讓介面更開闊
)

st.title("🤖 機器學習 RAG 分析專家")

# 1. 將技術指導原則移到側邊欄或折疊區，讓主介面更乾淨
with st.sidebar:
    st.header("⚙️ 專業設置與指導原則")
    st.markdown("這個機器人嚴格遵循下列技術指導，以提供專業且精準的答案：")
    
    # 使用折疊區顯示詳細的技術定義
    with st.expander("🛠️ 核心技術指導原則 (點擊查看)"):
        # 這裡直接引用 get_answer 函數中的 technical_guidance 內容 (假設您將它移到了全域或作為參數傳入)
        # 為了讓程式碼可執行，這裡簡化為重新定義，實際部署時請確保變數可訪問
        classification_key_terms = [
            "SVM：核心概念是尋找一個最大邊界 (Maximum Margin) 的超平面...",
            "當回答關於分類器的問題時，請先解釋該模型的**核心原理**與**關鍵參數**...", 
            # ... (這裡應該放入完整的 classification_key_terms 列表內容)
        ]
        st.code('\n'.join(classification_key_terms[:2]) + '\n...') # 顯示部分內容

    st.markdown("---")
    st.markdown("🌐 **Powered by Gemini & LangChain**")


# 2. 建立主要的問答容器
st.header("🤔 提問區")
st.write("請輸入您的機器學習分類器問題，例如：如何避免隨機森林的過擬合？")

# 使用 st.form 來美化輸入區塊和按鈕
with st.form("question_form", clear_on_submit=True):
    user_question = st.text_area(
        "問題輸入：",
        placeholder="請輸入關於 SVM、隨機森林、或模型評估指標的專業問題...",
        height=100
    )
    submit_button = st.form_submit_button("取得專業分析 🚀")

# 3. 處理點擊與輸出
if submit_button:
    if user_question:
        with st.spinner("🧠 機器人正在分析文件中，請稍候..."):
            # 確保 retriever 和 llm 已載入
            retriever = load_retriever()
            llm = load_llm()
            
            response = get_answer(user_question, retriever, llm)
            
            # 使用 st.container 和 st.success 讓結果更醒目
            st.subheader("✅ 專業回答")
            st.info(response)
            
            # 可選：新增一個下載按鈕
            # st.download_button(label="下載回答", data=response, file_name="analysis_result.txt")
    else:
        st.warning("請先輸入問題！")

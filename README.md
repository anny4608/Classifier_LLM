# 🤖 RAG-based Classifier Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about machine learning classifiers based on a provided set of PDF documents. It leverages Google's Gemini model for generation and HuggingFace embeddings with FAISS for efficient document retrieval. The application can be run as a local Python script or deployed as an interactive web application using Streamlit.

## 🤖 Demo site(因為有使用gemini api，api，所以學期結束就會關閉)
https://classifierllm.streamlit.app/
## Features

*   **Document Ingestion:** Loads and processes PDF documents from a specified directory.
*   **Vector Store Creation:** Utilizes HuggingFace embeddings to convert document chunks into vector representations and stores them in a FAISS index for fast similarity search.
*   **RAG-based Question Answering:** Combines retrieved document snippets with a predefined technical guidance and a user's query to generate informed answers using the Gemini 2.5 Flash model.
*   **Local Script:** A command-line interface (`app.py`) for testing the RAG functionality.
*   **Streamlit Web App:** An interactive web interface (`streamlit_app.py`) for a user-friendly experience.

## 🤖 Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.8+** installed.
*   **Google API Key:** Obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey). This key will be used to access the Gemini model.
*   **Git LFS (Large File Storage):** If your `faiss_index/index.faiss` file is expected to be large (over 100MB), you will need Git LFS to manage it in your GitHub repository for Streamlit deployment. Install it from [git-lfs.github.com](https://git-lfs.github.com/).

## 🤖 Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # On Windows
    source venv/bin/activate # On macOS/Linux

    pip install -r requirements.txt
    ```

3.  **Prepare the Data:**
    The project uses PDF lecture notes for "資料科學與回歸分析講義". You need to extract the contents of `資料科學與回歸分析講義.7z` into a specific directory structure.

    *   Create a folder named `books` in the root of your project directory.
    *   Extract the contents of `資料科學與回歸分析講義.7z` into a subfolder named `資料科學與回歸分析講義` inside the `books` folder.
    *   The final structure should look like:
        ```
        your-project/
        ├── books/
        │   └── 資料科學與回歸分析講義/
        │       ├── your_pdf_file_1.pdf
        │       └── your_pdf_file_2.pdf
        ├── app.py
        ├── streamlit_app.py
        ├── requirements.txt
        └── ...
        ```

4.  **Generate FAISS Vector Store:**
    The `app.py` script will generate the `faiss_index` directory containing your vector store. Run it once to create the index:
    ```bash
    python app.py
    ```
    This will create a `faiss_index` directory in your project root. This step is crucial before running the Streamlit app or deploying.

5.  **Set Google API Key:**
    Set your Google API Key as an environment variable.
    *   **On Windows:**
        ```bash
        set GOOGLE_API_KEY=YOUR_API_KEY
        ```
    *   **On macOS/Linux:**
        ```bash
        export GOOGLE_API_KEY=YOUR_API_KEY
        ```
    Replace `YOUR_API_KEY` with your actual key.

## 🤖 Usage

### Local Command-Line Application (`app.py`)

To test the RAG functionality directly from the command line:

```bash
python app.py
```
This will run an example query and print the response to your console.

### Local Streamlit Web Application (`streamlit_app.py`)

To run the interactive web application locally:

```bash
streamlit run streamlit_app.py
```
This command will open the Streamlit app in your web browser, usually at `http://localhost:8501`.

## 🤖 Deployment to Streamlit Community Cloud

To deploy your chatbot as a public web application:

1.  **GitHub Repository:**
    Ensure your project (including `streamlit_app.py`, `requirements.txt`, and the `faiss_index` directory) is pushed to a **public** GitHub repository. If `faiss_index/index.faiss` is large, ensure you've used Git LFS.

2.  **Streamlit Community Cloud:**
    *   Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
    *   Click on "**New app**".
    *   Select your GitHub repository, the branch (e.g., `main`), and set the "Main file path" to `streamlit_app.py`.

3.  **Configure Secrets:**
    *   Before deploying, click on "**Advanced settings**".
    *   In the "**Secrets**" section, add your `GOOGLE_API_KEY` in the following format:
        ```
        GOOGLE_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY"
        ```
    *   Replace `"YOUR_ACTUAL_GOOGLE_API_KEY"` with your key.

4.  **Deploy:**
    Click the "**Deploy!**" button. Streamlit will build and deploy your application, providing you with a public URL.

## File Structure

```
.
├── app.py                  # Local command-line RAG script
├── streamlit_app.py        # Streamlit web application
├── requirements.txt        # Python dependencies
├── faiss_index/            # Directory containing the FAISS vector store
│   ├── index.faiss
│   └── index.pkl
├── books/                  # Directory for PDF documents
│   └── 資料科學與回歸分析講義/
│       └── ... (your PDF files)
├── 用_RAG_打造分類器機器人.ipynb # Original Jupyter Notebook
├── 資料科學與回歸分析講義.7z   # Compressed PDF documents
└── README.md               # Project README file
```

## 🤖 Acknowledgements

*   The PDF lecture notes "資料科學與回歸分析講義" are used as the knowledge base for this chatbot.
*   Built with LangChain, Streamlit, and Google Gemini.

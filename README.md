# 🕸️ Dynamic Text-to-Graph RAG (Ollama & Neo4j)

A powerful **Streamlit** application that turns unstructured text into interactive knowledge graphs using **Local LLMs (Ollama)**.

## 🚀 Features

-   **📄 Text-to-Knowledge Graph**: Automatically extracts entities and relationships from any text input.
-   **🤖 Local LLM Privacy**: Uses **Ollama** (e.g., Llama 3) running locally for zero-data-leakage extraction.
-   **🔌 Hybrid Modes**:
    -   **Offline Mode**: Run entirely in memory without a database.
    -   **Neo4j Integration**: Sync extracted graphs to a Neo4j database for persistent storage and complex querying.
-   **🕸️ Interactive Visualization**:
    -   Visualize graphs using `streamlit-agraph`.
    -   Drag, zoom, and explore nodes dynamically.
-   **💬 Graph RAG Chat**:
    -   Chat with your graph!
    -   Uses **GraphRAG** (Retrieval Augmented Generation) to answer questions based *only* on the extracted relationships.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit, Streamlit-Agrah (Visualization)
-   **AI/LLM**: LangChain, Ollama (Llama 3)
-   **Database**: Neo4j (Optional)
-   **Backend Logic**: Python

## 📦 Installation

1.  **Clone the command**:
    ```bash
    git clone <your-repo-url>
    cd text_to_graph
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Ollama**:
    -   Download [Ollama](https://ollama.com).
    -   Pull the model: `ollama pull llama3` (or your preferred model).

## 🏃 Usage

1.  Start the app:
    ```bash
    streamlit run app.py
    ```
2.  **Offline Mode**: Just enter text and click "Process".
3.  **Neo4j Mode**: Toggle the sidebar switch, enter your Neo4j credentials, and connect.

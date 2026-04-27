# 🧠 RAG Chatbot Project
The RAG Chatbot Project is a Python-based application that utilizes the LangChain library and DeepSeek LLM to provide a conversational interface for users. The project's primary purpose is to load a vector database, set up a retriever, and define a prompt template for user interactions. The chatbot is designed to generate responses to user queries by combining the retriever, prompt, and LLM.

# 🚀 Features
Vector Database Management: The project uses the LangChain library to manage a vector database, which stores vector embeddings of documents.
Retriever: The project sets up a retriever object that searches the vector database for similar documents based on user input.
Prompt Template: The project defines a prompt template that structures user interactions, including context and question.
LLM Integration: The project integrates the DeepSeek LLM model to generate responses to user queries.
Gradio Chat Interface: The project includes a Gradio chat interface that invokes the RAG chain to generate responses to user input.
🛠️ Tech Stack
LangChain Library: The project uses the LangChain library for vector database management, prompt templates, and LLM integration.
DeepSeek LLM: The project uses the DeepSeek LLM model for generating responses to user queries.
Gradio: The project uses Gradio for building the chat interface.
Python: The project is built using Python as the primary programming language.
SQLite: The project uses SQLite as the database management system for storing vector embeddings.
# 📦 Installation
To install the project dependencies, run the following command:

pip install -r requirements.txt
This will install the required libraries and their versions specified in the requirements.txt file.

# 💻 Usage
To run the project, follow these steps:

Create the vector database by running the create_db.py file:
python src/create_db.py
Run the rag_app.py file to start the chat interface:
python src/rag_app.py
This will launch the Gradio chat interface, and you can interact with the chatbot by typing queries and receiving responses.

# 📂 Project Structure
```text
.
├── chroma_db
│   └── chroma.sqlite3
├── src
│   ├── create_db.py
│   ├── rag_app.py
│   └── __init__.py
├── requirements.txt
└── README.md
```

# 📬 Contact
For any questions, concerns, or feedback, please contact us at 2033079756@qq.com.

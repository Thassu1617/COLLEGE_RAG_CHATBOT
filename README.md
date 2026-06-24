College RAG Chatbot
An AI-powered College Information Chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from college documents. The application processes PDF files, converts them into vector embeddings using Hugging Face models, stores them in a FAISS vector database, and retrieves the most relevant information based on user queries through an interactive Flask web interface.

🚀 Features
📄 PDF document ingestion and processing

🔍 Semantic search using FAISS Vector Store

🤖 AI-powered question answering

🧠 Hugging Face Sentence Transformer embeddings

🌐 Interactive Flask web application

⚡ Fast and accurate information retrieval

📚 College information assistant for admissions, courses, fees, placements, facilities, and more

🛠️ Tech Stack
Python

Flask

LangChain

FAISS

Hugging Face Transformers

Sentence Transformers

HTML

CSS

JavaScript

📂 Project Structure
College_RAG_Chatbot/
│
├── app.py                 # Flask application
├── ingest.py              # PDF processing and vector database creation
├── college_info.pdf       # Knowledge source document
├── college_db/            # FAISS vector database
│
├── templates/
│   └── index.html         # Frontend UI
│
└── README.md
⚙️ Installation
1. Clone Repository
git clone https://github.com/your-username/College_RAG_Chatbot.git
cd College_RAG_Chatbot
2. Install Dependencies
pip install flask
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-text-splitters
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
📥 Create Vector Database
Place your PDF file in the project folder and run:

python ingest.py
This will:

Load the PDF document

Split text into chunks

Generate embeddings

Store embeddings in a FAISS vector database

▶️ Run the Application
python app.py
Open your browser:

http://127.0.0.1:5000
💡 Example Questions
What courses are offered by the college?

What is the admission process?

What are the placement statistics?

What facilities are available on campus?

What are the tuition fees?

Tell me about the departments.

🎯 How It Works
PDF documents are loaded and processed.

Text is split into smaller chunks.

Hugging Face embeddings convert text into vectors.

FAISS stores vectors for efficient retrieval.

User submits a question.

Similarity search retrieves relevant content.

The chatbot returns the most relevant answer.

📈 Future Enhancements
Integration with OpenAI/Gemini LLMs

Multi-PDF support

Chat history storage

Voice-based interaction

Dark/Light theme support

Advanced RAG pipeline with re-ranking

Deployment on Render, Railway, or AWS

👨‍💻 Author
PALAGIRI THASNEEM

AI-powered College Information Chatbot using Flask, LangChain, FAISS, and Hugging Face Embeddings for intelligent document-based question answering.

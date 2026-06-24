from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

app = Flask(__name__)

# 1. Load Embeddings and Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("college_db", embeddings, allow_dangerous_deserialization=True)

# 2. Load the Text Generation Model 
# (Using flan-t5-base as it is lightweight and great for conversational QA)
llm = pipeline("text2text-generation", model="google/flan-t5-base")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json["question"]
        
        # Retrieve relevant documents
        docs = db.similarity_search(question, k=2)
        if not docs:
            return jsonify({"answer": "I'm sorry, I couldn't find any relevant information about that in my database."})
            
        # Combine the retrieved context
        context = " ".join([doc.page_content for doc in docs])
        
        # Create a prompt combining the question and the context for the LLM
        prompt = f"Use the following information to answer the user's question in a clear, natural, and conversational way.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Generate the conversational response
        response = llm(prompt, max_new_tokens=150)
        conversational_answer = response[0]['generated_text']
        
        return jsonify({"answer": conversational_answer})
        
    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    # Listening on 7860 so Hugging Face Spaces can run it!
    app.run(host="0.0.0.0", port=7860, debug=False)

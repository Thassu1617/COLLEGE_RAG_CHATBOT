from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "college_db",
    embeddings,
    allow_dangerous_deserialization=True
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json["question"]

        docs = db.similarity_search(question, k=2)

        if not docs:
            return jsonify({
                "answer": "No relevant information found."
            })

        answer = docs[0].page_content

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        return jsonify({
            "answer": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

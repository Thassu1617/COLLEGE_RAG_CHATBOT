from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Initialize Flask Application
app = Flask(__name__)

# 2. Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Load Vector Database
db = FAISS.load_local(
    "college_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# 4. Initialize Local Offline Model Engine
print("⏳ Loading local offline model engine...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
print("✅ System initialized completely OFFLINE.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data["question"]
        question_lower = question.lower().strip()

        # Search database for context chunks
        docs_with_scores = db.similarity_search_with_score(question, k=2)
        
        if not docs_with_scores:
            return jsonify({"answer": "I don't have information on that topic."})

        best_score = docs_with_scores[0][1]

        # Target bypass keywords
        college_keywords = ["fee", "fees", "syllabus", "placement", "placements", "admission", "admissions", "course", "courses"]
        has_keyword = any(kw in question_lower for kw in college_keywords)

        # GUARDRAIL
        if best_score > 1.7 and not has_keyword:
            return jsonify({
                "answer": "🤖 I am only programmed to answer questions directly related to our college admissions, courses, fees, syllabus, and campus placements. Please ask a more detailed question!"
            })

        # Combine text from verified database chunks
        context = " ".join([doc.page_content.strip() for doc, score in docs_with_scores])

        # 5. Native Qwen Chat Format with ultra-strict filtering instructions
        prompt = (
            f"<|im_start||system\n"
            f"You are a direct college assistant. Answer the user's question in exactly 1 or 2 clear sentences using ONLY the facts provided.\n"
            f"CRITICAL: Do not mention or include any extra, unrelated facts from the text that the user did not explicitly ask for.\n"
            f"Facts:\n{context}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # Local Safe Generation
        res = qa_pipeline(
            prompt, 
            max_new_tokens=100, # Reduced to keep answers short and focused
            temperature=0.1, 
            do_sample=False
        )
        full_text = res[0]['generated_text']

        # Extract only the freshly generated assistant text
        if "<|im_start|>assistant\n" in full_text:
            answer = full_text.split("<|im_start|>assistant\n")[-1].strip()
        else:
            answer = full_text.replace(prompt, "").strip()

        # Clean off any trailing tokens or format markers
        clean_answer = answer.replace("<|im_end|>", "").strip()

        return jsonify({"answer": clean_answer})

    except Exception as e:
        return jsonify({"answer": f"Error formatting offline response: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
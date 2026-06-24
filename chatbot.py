from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# 1. Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Vector Database
db = FAISS.load_local(
    "college_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Load Text Generation Pipeline (Matching the Flask backend)
print("⏳ Loading local AI text engine (Qwen)... Please wait.")
qa_pipeline = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    max_new_tokens=150
)

print("=" * 60)
print("🎓 COLLEGE AI ASSISTANT (CONVERSATIONAL)")
print("=" * 60)
print("Ask anything about admissions, fees, courses, or placements.")
print("Type 'exit' to quit")
print("=" * 60)

while True:
    query = input("\n🧑 You: ").strip()

    if not query:
        continue

    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    # 4. Search with L2 distance scores for topic validation
    docs_with_scores = db.similarity_search_with_score(query, k=2)

    print("\n🤖 College Assistant: ", end="", flush=True)

    if not docs_with_scores:
        print("I don't have information on that topic.")
        continue

    # Extract the top matching distance score
    best_score = docs_with_scores[0][1]

    # Guardrail: Prevent the bot from talking about unrelated matters
    if best_score > 1.3:
        print("I am only programmed to answer questions directly related to our college admissions, courses, fees, and campus placements. Please ask a relevant question!")
        continue

    # Combine facts from verified semantic hits
    context = " ".join([doc.page_content.strip() for doc, score in docs_with_scores])

    # 5. Mirror the strict system layout structure
    prompt = (
        f"<system>You are a strict, dedicated college help-desk representative.\n"
        f"RULE 1: Answer the query using ONLY the provided facts below.\n"
        f"RULE 2: If the question cannot be answered completely using ONLY the provided facts, "
        f"respond exactly with: 'I am sorry, but I only possess information regarding college operations, fees, and placements.'\n"
        f"Do not use your own general external knowledge.</system>\n"
        f"<facts>{context}</facts>\n"
        f"User: {query}\n"
        f"Assistant:"
    )

    # Generate response
    res = qa_pipeline(prompt, temperature=0.0, do_sample=False)
    full_text = res[0]['generated_text']

    # Isolate assistant block execution
    if "Assistant:" in full_text:
        answer = full_text.split("Assistant:")[-1].strip()
    else:
        answer = full_text.replace(prompt, "").strip()

    # Clean up lagging strings or artifact tags
    clean_answer = answer.split("User:")[0].replace("</system>", "").strip()

    print(clean_answer)
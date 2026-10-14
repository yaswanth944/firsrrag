import re
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

PERSIST_DIR = "db/chroma_db"
MODEL_NAME = "qwen2.5:7b"   # run: ollama pull qwen2.5:7b  (1.5b is too weak to follow grounding rules)
K = 4
THRESHOLD = 0.35            # TUNE THIS using the printed scores (see steps in the guide)
REFUSAL = "I don't have enough information to answer that question based on the provided documents."

SYSTEM_PROMPT = f"""You are a strict question-answering assistant.
Use ONLY the documents provided. Never use outside knowledge. Never guess.
A document counts only if it explicitly answers the exact question, about the exact subject asked.
Facts about one subject (e.g. a Microsoft CEO) must never be used to answer a question about a different subject (e.g. India's prime minister).

Reply in EXACTLY this format:
QUOTE: <copy the one sentence/line from the documents that answers the question, word for word>
ANSWER: <short answer based only on that quote>

If no document explicitly answers the question, reply with exactly:
{REFUSAL}"""


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def main(query):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 1) Retrieve WITH scores and print them so you can tune THRESHOLD
    results = db.similarity_search_with_relevance_scores(query, k=K)
    print(f"\nUser Query: {query}")
    print("--- Retrieval scores (higher = more similar) ---")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:70].replace("\n", " ")
        print(f"{i}. {score:.3f} | {preview}")

    # 2) Relevance gate: if nothing is relevant, never call the LLM
    good = [(d, s) for d, s in results if s >= THRESHOLD]
    if not good:
        print("\n--- Response ---")
        print(REFUSAL + "  (blocked by score gate, LLM not called)")
        return

    docs_text = "\n\n".join(f"[Doc {i}] {d.page_content}" for i, (d, _) in enumerate(good, 1))
    user_msg = f"""Documents:
{docs_text}

Question: {query}

Follow the required format. If the documents do not explicitly answer this exact question, give the refusal sentence."""

    # 3) Generate
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    result = model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    raw = result.content.strip()

    # 4) Verify the quote really exists in the retrieved context
    print(f"\n--- Response (model: {MODEL_NAME}) ---")
    if REFUSAL.lower() in raw.lower():
        print(REFUSAL)
        return

    quote_match = re.search(r"QUOTE:\s*(.+?)\s*(?:\n\s*ANSWER:|$)", raw, re.S | re.I)
    answer_match = re.search(r"ANSWER:\s*(.+)", raw, re.S | re.I)
    context_norm = normalize(docs_text)

    if quote_match and answer_match and normalize(quote_match.group(1).strip('"\'')) in context_norm:
        print(answer_match.group(1).strip())
        print(f"\n(Supporting quote verified: \"{quote_match.group(1).strip()}\")")
    else:
        print(REFUSAL)
        print("(Rejected: the model's quote was missing or not found in the documents.)")
        print("Raw model output:", raw)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "who is the current prime minister of India?"
    main(q)
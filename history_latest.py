import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

PERSIST_DIR = "db/chroma_db"
MODEL_NAME = "qwen2.5:7b"
K = 4
THRESHOLD = 0.35          # tune from printed scores
MAX_HISTORY_TURNS = 4
REFUSAL = "I don't have enough information to answer that question based on the provided documents."

REWRITE_PROMPT = """You turn the user's latest message into a standalone search query.
Rules:
1. Only fix references that depend on earlier messages (it, that, they, "and what about...", missing subject).
2. KEEP the user's own key words. Do NOT swap them for different words and do NOT add new facts.
3. If the message already makes sense on its own, return it UNCHANGED.
4. Never answer the question. Never repeat the previous question unless the user did.
Output ONLY the query, one line."""

ANSWER_PROMPT = f"""You are a strict question-answering assistant.
Use ONLY the Documents for facts. Chat history is ONLY for understanding what the question refers to; never take facts from it.
A document counts only if it explicitly answers the exact question about the exact subject asked.

Reply in EXACTLY this format:
QUOTE: <one sentence/line copied word for word from the Documents>
ANSWER: <short answer based only on that quote>

If no document explicitly answers, reply with exactly:
{REFUSAL}"""


def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()


def rewrite_query(query, history, llm):
    if not history:
        return query                      # first turn: nothing to resolve
    convo = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history[-MAX_HISTORY_TURNS:])
    msg = f"Conversation so far:\n{convo}\n\nLatest user message: {query}\n\nStandalone search query:"
    out = llm.invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=msg)]).content
    out = out.strip().split("\n")[0].strip().strip("\"'")
    if not out or len(out.split()) > 3 * max(len(query.split()), 4) + 8:
        return query                      # reject empty or runaway rewrites
    return out


def retrieve(db, queries):
    """Search with every query variant, merge duplicates, keep the best score per chunk."""
    pool = {}
    for q in queries:
        for doc, score in db.similarity_search_with_relevance_scores(q, k=K):
            if doc.page_content not in pool or score > pool[doc.page_content][1]:
                pool[doc.page_content] = (doc, score)
    return sorted(pool.values(), key=lambda x: x[1], reverse=True)[:K]


def answer(query, history, last_docs, db, llm):
    rewritten = rewrite_query(query, history, llm)
    queries = [query] if rewritten.lower() == query.lower() else [query, rewritten]
    print(f"Search queries: {queries}")

    ranked = retrieve(db, queries)
    print("--- Retrieval scores ---")
    for i, (d, s) in enumerate(ranked, 1):
        print(f"{i}. {s:.3f} | {d.page_content[:70]!r}")

    good = [d for d, s in ranked if s >= THRESHOLD]

    # Follow-up question: also keep the chunks that answered the previous turn
    is_followup = len(queries) > 1
    if is_followup:
        for d in last_docs:
            if d not in good:
                good.append(d)

    if not good:
        return REFUSAL, []

    docs_text = "\n\n".join(f"[Doc {i}] {d.page_content}" for i, d in enumerate(good, 1))
    convo = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history[-MAX_HISTORY_TURNS:]) or "(none)"
    user_msg = (f"Chat history:\n{convo}\n\nDocuments:\n{docs_text}\n\n"
                f"Question: {query}\n(Standalone form: {rewritten})\n\n"
                "Follow the required format. If the Documents do not explicitly answer it, give the refusal sentence.")

    raw = llm.invoke([SystemMessage(content=ANSWER_PROMPT), HumanMessage(content=user_msg)]).content.strip()

    if REFUSAL.lower() in raw.lower():
        return REFUSAL, good
    q = re.search(r"QUOTE:\s*(.+?)\s*(?:\n\s*ANSWER:|$)", raw, re.S | re.I)
    a = re.search(r"ANSWER:\s*(.+)", raw, re.S | re.I)
    if q and a and norm(q.group(1).strip("\"'")) in norm(docs_text):
        return a.group(1).strip(), good
    print(f"(Rejected unverified output: {raw!r})")
    return REFUSAL, good


def main():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb,
                collection_metadata={"hnsw:space": "cosine"})
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    history, last_docs = [], []
    print(f"History-aware RAG ({MODEL_NAME}). Type 'quit' to exit, 'reset' to clear history.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "reset":
            history, last_docs = [], []
            print("History cleared.")
            continue
        if not query:
            continue
        reply, docs = answer(query, history, last_docs, db, llm)
        print(f"\nAnswer: {reply}")
        history.append((query, reply))
        last_docs = docs


if __name__ == "__main__":
    main()
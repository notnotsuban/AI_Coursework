import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL          = os.getenv("CHAT_MODEL", "llama3.2:latest")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH      = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TOP_K               = int(os.getenv("NAIVE_TOP_K", 8))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about college regulations.
Answer ONLY based on the provided context. If the context does not contain enough information, say so.
Be concise, accurate, and cite relevant rules or sections when possible."""


class NaiveRAG:
    def __init__(self):
        self.embedder  = SentenceTransformer(EMBEDDING_MODEL)
        self.client_db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client_db.get_collection("naive_rag")
        # Inside naive_rag.py and advanced_rag.py
        # Change from localhost:11434 to your server IP
        self.llm = OpenAI(
            base_url="http://192.168.1.81:11434/v1",
            api_key="ollama",
            timeout=120
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Embed query, do cosine-similarity search, return top-k chunks."""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":     doc,
                "metadata": meta,
                "score":    1 - dist,  # cosine similarity
            })
        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        """Concatenate retrieved chunks into a single context string."""
        parts = []
        for i, c in enumerate(chunks, 1):
            source = c["metadata"].get("source", "unknown")
            page   = c["metadata"].get("page_start", "?")
            parts.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        """Send context + query to LLM, return answer."""
        user_message = f"""Context from college regulations:

{context}

Question: {query}

Answer:"""
        response = self.llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def query(self, question: str) -> dict:
        """Full naive RAG pipeline. Returns answer + retrieved context."""
        chunks  = self.retrieve(question)
        context = self.build_context(chunks)
        answer  = self.generate(question, context)
        return {
            "question": question,
            "answer":   answer,
            "contexts": [c["text"] for c in chunks],
            "scores":   [c["score"] for c in chunks],
            "metadata": [c["metadata"] for c in chunks],
        }
def run_naive_rag(question):
    rag = NaiveRAG()
    result = rag.query(question)
    
    # Return the answer string and the context list
    return result['answer'], result['contexts']

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = NaiveRAG()
    q   = "What are the attendance requirements for students?"
    result = rag.query(q)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print(f"Retrieved {len(result['contexts'])} chunks")
    for i, (ctx, score) in enumerate(zip(result["contexts"], result["scores"]), 1):
        print(f"\n  Chunk {i} (score={score:.3f}):\n  {ctx[:200]}...")
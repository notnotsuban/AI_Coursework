import os
import json
import math
import re
from pathlib import Path

import chromadb
import nltk
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL          = os.getenv("CHAT_MODEL", "llama3.2:latest")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH      = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TOP_K               = int(os.getenv("ADVANCED_TOP_K", 8))
RERANK_TOP_K        = int(os.getenv("ADVANCED_RERANK_TOP_K", 5))

SYSTEM_PROMPT = """You are a precise assistant that answers questions about college regulations.
Answer ONLY based on the provided context. If the context does not contain enough information, say so.
Be concise, accurate, and cite relevant rules or sections when possible."""


class AdvancedRAG:
    def __init__(self):
        print("  [AdvancedRAG] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        print("  [AdvancedRAG] Loading cross-encoder re-ranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("  [AdvancedRAG] Connecting to ChromaDB...")
        self.client_db  = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client_db.get_collection("advanced_rag")

        print("  [AdvancedRAG] Loading BM25 corpus...")
        corpus_path = Path(CHROMA_DB_PATH) / "bm25_corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_texts = json.load(f)
        
        # FIXED: Ensure we are tokenizing the "text" field of the corpus objects
        tokenized = [self._tokenize(t) for t in corpus_texts]
        self.bm25          = BM25Okapi(tokenized)
        self.bm25_corpus   = corpus_texts

        # Pointing to your Ubuntu server
        self.llm = OpenAI(
            base_url="http://192.168.1.81:11434/v1",
            api_key="ollama", 
            timeout=120
        )
        print("  [AdvancedRAG] Ready.\n")

    # ── Step 1: Query rewriting ──────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """Ask the LLM to reformulate the query for better document retrieval."""
        prompt = f"""You are a query optimization expert. Rewrite the following question to be more specific and retrieval-friendly for searching through college regulation documents. Return ONLY the rewritten query, nothing else.

Original query: {query}
Rewritten query:"""
        try:
            resp = self.llm.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            rewritten = resp.choices[0].message.content.strip()
            if len(rewritten) < 5 or len(rewritten) > 300:
                return query
            return rewritten
        except Exception:
            return query

    # ── Step 2: HyDE (Hypothetical Document Embedding) ──────────────────────

    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer to the query for embedding."""
        prompt = f"""Write a short, plausible passage (2-3 sentences) from a college regulation document that would answer this question. Write ONLY the passage, no preamble.

Question: {query}
Passage:"""
        try:
            resp = self.llm.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return query 

    # ── Step 3: Hybrid retrieval ─────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()

    def _dense_retrieve(self, text_to_embed: str, top_k: int) -> list[dict]:
        """Dense retrieval via ChromaDB cosine similarity."""
        emb = self.embedder.encode(text_to_embed).tolist()
        results = self.collection.query(
            query_embeddings=[emb],
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
                "dense_score": 1 - dist,
            })
        return chunks

    def _sparse_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Sparse retrieval via BM25."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:
                # FIXED: raw_chunk is a string, so we can't call .items()
                raw_chunk_text = self.bm25_corpus[idx]
                
                chunks.append({
                    "text":         raw_chunk_text,
                    "metadata":     {}, # No metadata available in a string list
                    "sparse_score": float(scores[idx]),
                })
        return chunks

    def hybrid_retrieve(self, query: str, hyde_text: str, top_k: int) -> list[dict]:
        """Combine dense + sparse results using RRF."""
        dense_results  = self._dense_retrieve(hyde_text, top_k * 2)
        sparse_results = self._sparse_retrieve(query, top_k * 2)

        k = 60 
        scores: dict[str, float] = {}
        text_map: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results, 1):
            key = hash(chunk["text"])
            scores[key]   = scores.get(key, 0) + 1 / (k + rank)
            text_map[key] = chunk

        for rank, chunk in enumerate(sparse_results, 1):
            key = hash(chunk["text"])
            scores[key]   = scores.get(key, 0) + 1 / (k + rank)
            if key not in text_map:
                text_map[key] = chunk

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        fused = []
        for key in sorted_keys:
            c = text_map[key].copy()
            c["rrf_score"] = scores[key]
            fused.append(c)

        return fused

    # ── Step 4: Re-ranking ─────────────────────────────────────

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return chunks
        pairs  = [(query, c["text"]) for c in chunks]
        scores = self.reranker.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    # ── Step 5: Contextual compression ───────────────────────────────────────

    def compress_chunk(self, query: str, chunk_text: str) -> str:
        """Extract only relevant sentences."""
        prompt = f"""From the following passage, extract ONLY the sentences that are directly relevant to answering the question. If nothing is relevant, return the first two sentences. Return only the extracted text, no commentary.

Question: {query}

Passage:
{chunk_text}

Relevant sentences:"""
        try:
            resp = self.llm.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            compressed = resp.choices[0].message.content.strip()
            return compressed if len(compressed) < len(chunk_text) else chunk_text
        except Exception:
            return chunk_text

    # ── Step 6: Generation ───────────────────────────────────────────────────

    def build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            meta    = c.get("metadata", {})
            source  = meta.get("source", "unknown")
            page    = meta.get("page", meta.get("page_start", "?"))
            section = meta.get("section_hint", "")
            header  = f"[Chunk {i} | Source: {source} | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += "]"
            parts.append(f"{header}\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
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

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Full Advanced RAG pipeline.
        Returns a dictionary containing the answer and contexts.
        """
        steps = {}

        # 1. Query rewriting
        rewritten = self.rewrite_query(question)
        steps["rewritten_query"] = rewritten

        # 2. HyDE
        hyde_text = self.generate_hypothetical_answer(rewritten)
        steps["hyde_text"] = hyde_text

        # 3. Hybrid retrieval
        chunks = self.hybrid_retrieve(rewritten, hyde_text, TOP_K)
        steps["retrieved_count"] = len(chunks)

        # 4. Re-ranking
        chunks = self.rerank(rewritten, chunks, RERANK_TOP_K)

        # 5. Contextual compression
        for c in chunks:
            c["text"] = self.compress_chunk(rewritten, c["text"])

        # 6. Build final context and generate answer
        context = self.build_context(chunks)
        answer  = self.generate(question, context)

        return {
            "answer":   answer,
            "contexts": [c["text"] for c in chunks],
            "question": question,
            "scores":   [c.get("rerank_score", c.get("rrf_score", 0.0)) for c in chunks],
            "metadata": [c.get("metadata", {}) for c in chunks],
            "steps":    steps,
        }

def run_advanced_rag(question):
    rag = AdvancedRAG() 
    
    # Run the query logic
    result = rag.query(question)
    
    return result['answer'], result['contexts']

if __name__ == "__main__":
    rag    = AdvancedRAG()
    q      = "What are the attendance requirements for students?"
    result = rag.query(q, verbose=True)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
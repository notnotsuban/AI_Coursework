import os
import re
import json
import nltk
import fitz  # PyMuPDF
import chromadb
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Download NLTK data quietly ─────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_FOLDER       = os.getenv("DATA_FOLDER", "./data")
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
NAIVE_CHUNK_SIZE  = int(os.getenv("NAIVE_CHUNK_SIZE", 500))
NAIVE_OVERLAP     = int(os.getenv("NAIVE_CHUNK_OVERLAP", 50))
ADV_CHUNK_SIZE    = int(os.getenv("ADVANCED_CHUNK_SIZE", 300))
ADV_OVERLAP       = int(os.getenv("ADVANCED_CHUNK_OVERLAP", 100))


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF, returning list of {page, text}."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text = re.sub(r'\n{3,}', '\n\n', text)   # collapse excessive newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse spaces
        text = text.strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def naive_chunk(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Fixed-size character chunking across all pages.
    Returns list of {chunk_id, text, source, page_start}.
    """
    full_text = ""
    page_boundaries = []  # (start_char, page_num)
    for p in pages:
        page_boundaries.append((len(full_text), p["page"]))
        full_text += p["text"] + "\n\n"

    chunks = []
    start = 0
    idx   = 0
    while start < len(full_text):
        end  = min(start + chunk_size, len(full_text))
        text = full_text[start:end].strip()
        if text:
            # figure out which page this chunk starts on
            page_num = 1
            for boundary_start, pnum in page_boundaries:
                if boundary_start <= start:
                    page_num = pnum
            chunks.append({
                "chunk_id":   f"naive_{idx}",
                "text":       text,
                "page_start": page_num,
            })
            idx += 1
        start += chunk_size - overlap

    return chunks


def advanced_chunk(pages: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Sentence-aware semantic chunking.
    Groups sentences until chunk_size is reached, then starts a new chunk
    with `overlap` characters of carry-over context.
    Returns list of {chunk_id, text, source, page, section_hint}.
    """
    chunks = []
    idx    = 0

    for page_data in pages:
        page_num = page_data["page"]
        text     = page_data["text"]

        # Tokenise into sentences
        sentences = nltk.sent_tokenize(text)

        current_chunk   = []
        current_len     = 0
        carry_over_sents = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            current_chunk.append(sent)
            current_len += len(sent) + 1  # +1 for space

            if current_len >= chunk_size:
                chunk_text = " ".join(current_chunk).strip()
                # Detect a rough "section hint" from the first sentence
                section_hint = _detect_section(current_chunk[0])

                chunks.append({
                    "chunk_id":     f"adv_{idx}",
                    "text":         chunk_text,
                    "page":         page_num,
                    "section_hint": section_hint,
                })
                idx += 1

                # carry-over: keep last few sentences for context continuity
                carry_over_text = 0
                carry_over_sents = []
                for s in reversed(current_chunk):
                    carry_over_text += len(s)
                    carry_over_sents.insert(0, s)
                    if carry_over_text >= overlap:
                        break

                current_chunk = carry_over_sents
                current_len   = sum(len(s) for s in current_chunk)

        # flush remaining
        if current_chunk:
            chunk_text   = " ".join(current_chunk).strip()
            section_hint = _detect_section(current_chunk[0])
            chunks.append({
                "chunk_id":     f"adv_{idx}",
                "text":         chunk_text,
                "page":         page_num,
                "section_hint": section_hint,
            })
            idx += 1

    return chunks


def _detect_section(text: str) -> str:
    """Heuristically detect if a sentence looks like a section heading."""
    text = text.strip()
    # All-caps short line, or numbered heading like "1.2 Admission"
    if re.match(r'^(\d+[\.\d]*\s+[A-Z]|[A-Z\s]{5,40})$', text):
        return text[:60]
    return ""


def embed_and_store(
    collection,
    chunks: list[dict],
    embedder: SentenceTransformer,
    source_name: str,
    batch_size: int = 64,
):
    """Embed chunks in batches and upsert into a ChromaDB collection."""
    total = len(chunks)
    for i in tqdm(range(0, total, batch_size), desc=f"  Embedding → {collection.name}"):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids   = [f"{source_name}__{c['chunk_id']}" for c in batch]

        metadatas = []
        for c in batch:
            meta = {"source": source_name}
            for k, v in c.items():
                if k in ("text", "chunk_id"):
                    continue
                meta[k] = str(v) if not isinstance(v, (int, float, bool)) else v
            metadatas.append(meta)

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )


# ── Main ingestion pipeline ──────────────────────────────────────────────────

def ingest():
    print("=" * 60)
    print("  RAG Ingestion Pipeline")
    print("=" * 60)

    # 1. Load PDFs
    data_path = Path(DATA_FOLDER)
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_FOLDER}'")
    print(f"\n📄 Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    # 2. Load embedding model
    print(f"\n🔤 Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Init ChromaDB
    print(f"\n🗄️  Initialising ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collections to allow re-ingestion
    for col_name in ["naive_rag", "advanced_rag"]:
        try:
            client.delete_collection(col_name)
            print(f"   Deleted existing collection '{col_name}'")
        except Exception:
            pass

    naive_col = client.create_collection(
        name="naive_rag",
        metadata={"hnsw:space": "cosine"},
    )
    adv_col = client.create_collection(
        name="advanced_rag",
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Process each PDF
    all_naive_chunks   = []
    all_advanced_chunks = []
    corpus_texts       = []  # for BM25 later

    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        print(f"\n📖 Processing: {pdf_path.name}")

        pages = extract_text_from_pdf(str(pdf_path))
        print(f"   Pages extracted: {len(pages)}")

        # Naive chunks
        n_chunks = naive_chunk(pages, NAIVE_CHUNK_SIZE, NAIVE_OVERLAP)
        print(f"   Naive chunks   : {len(n_chunks)}")

        # Advanced chunks
        a_chunks = advanced_chunk(pages, ADV_CHUNK_SIZE, ADV_OVERLAP)
        print(f"   Advanced chunks: {len(a_chunks)}")

        # Tag source
        for c in n_chunks:
            c["source"] = source_name
        for c in a_chunks:
            c["source"] = source_name

        all_naive_chunks.extend(n_chunks)
        all_advanced_chunks.extend(a_chunks)
        # Preserve the full chunk dict (text + metadata) for hybrid search
        corpus_texts.extend(a_chunks)

        # Embed & store
        embed_and_store(naive_col, n_chunks, embedder, source_name)
        embed_and_store(adv_col,  a_chunks, embedder, source_name)

    # 5. Save BM25 corpus for advanced RAG (hybrid retrieval)
    corpus_path = Path(CHROMA_DB_PATH) / "bm25_corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_texts, f, ensure_ascii=False)
    print(f"\n💾 BM25 corpus saved: {corpus_path} ({len(corpus_texts)} docs)")

    print(f"\n✅ Ingestion complete!")
    print(f"   Naive   collection: {naive_col.count()} chunks")
    print(f"   Advanced collection: {adv_col.count()} chunks")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
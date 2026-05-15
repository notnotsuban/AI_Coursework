import os
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate

# Fix: Import the metrics from collections to address deprecation warnings
from ragas.metrics import faithfulness, answer_relevancy
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
import nltk

from naive_rag import NaiveRAG
from advanced_rag import AdvancedRAG

load_dotenv()
# ── Config ───────────────────────────────────────────────────────────────────
CHAT_MODEL  = os.getenv("CHAT_MODEL", "llama3.2:latest")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Queries ───────────────────────────────────────────────────────────────────

def get_custom_queries() -> list[str]:
    return [

    "What is the minimum IELTS score requirement for admission into postgraduate taught courses?",
    "How does the University treat applications from candidates with criminal convictions?",
    "Are international students on a Tier 4 visa subject to different English language criteria than standard applicants?",
    "Under what specific circumstances can the University terminate a student's registration?",
    "What happens to a student's visa status if they take a break from studies while holding a Tier 4 visa?",
    "How many authorised absences is a student allowed per semester for a single module?",
    "What are the potential recommendations a School Nominee can make during an Exploratory Interview for Fitness to Study?",
    "Can a student's registration be terminated purely on the grounds of Fitness to Study?",
    "What is the maximum number of attempts permitted for any single assessment?",
    "How is an Integrated Masters degree classification calculated in terms of credit weighting?",
    "What happens to a student's module mark if they fail but are granted an 'in-year reassessment'?",
    "What is an Aegrotat award, and what evidence is required to receive one?",
    "For a Postgraduate Masters degree, how many credits are required at Level 7?",
    "Is a student allowed to retake a module they have already passed to improve their grade?",
    "What is the strict deadline for submitting a claim for Mitigating Circumstances?",
    "What happens if a mitigating circumstances claim is rejected due to a lack of independent supporting evidence?",
    "Who is responsible for assessing whether mitigating circumstances claims meet the necessary criteria?",
    "On what specific grounds can a student submit a procedural defect appeal against an Assessment Board decision?",
    "Will the University consider an appeal based on a student disagreeing with the academic judgment of an examiner?",
    "What is the time limit for submitting a formal complaint to the University after an incident occurs?",
    "Which penalty applies to a student found commissioning another person to write their essay?",
    "What happens if a student refuses to respond to an academic misconduct allegation within the 10-day deadline?",
    "Under what precise circumstances can a previously closed academic misconduct allegation be reconsidered for a second time?",
    "Under the QAHE Student Code of Conduct, what are the strict rules regarding mobile phone use during lectures?",
    "What language are QAHE students expected to use exclusively when communicating with one another during seminars?",
    "What specific items are strictly prohibited from being brought onto the QAHE campus?",
    "According to the QAHE Code of Conduct, whose authority must be obeyed regarding the overall safety of everyone on campus?",
    "Who must initial non-academic misconduct allegations be reported to under the QAHE Student Code of Conduct?",
    "What are the four acceptable grounds for appealing a QAHE Misconduct Panel decision?",
    ]

# ── RAGAS evaluator setup ─────────────────────────────────────────────────────

def build_ragas_llm():
    client = OpenAI(base_url="http://192.168.1.81:11434/v1", api_key="ollama")
    return llm_factory(model="llama3.1:latest", client=client)

class CustomOllamaEmbeddings:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url="http://192.168.1.81:11434/v1", api_key="ollama")
        
    def embed_query(self, text: str) -> list[float]:
        res = self.client.embeddings.create(model="all-minilm", input=[text])
        return res.data[0].embedding
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = self.client.embeddings.create(model="all-minilm", input=texts)
        return [d.embedding for d in res.data]

# ── Run evaluation ────────────────────────────────────────────────────────────

def run_rag_on_queries(rag, queries: list[str], name: str) -> list[dict]:
    results = []
    print(f"\n🔄 Running {name} on {len(queries)} queries...")
    for q in tqdm(queries, desc=f"  {name}"):
        try:
            result = rag.query(q)
            # Handle cases where RAG might return a string or dict
            if isinstance(result, str):
                ans, ctx = result, []
            else:
                ans = result.get("answer", "No answer")
                ctx = result.get("contexts", [])
            results.append({"question": q, "answer": ans, "contexts": ctx})
        except Exception as e:
            print(f"  ⚠️ Error in {name}: {e}")
            results.append({"question": q, "answer": "Error", "contexts": []})
    return results

def evaluate_both():
    print("=" * 65)
    print("  RAG Evaluation Pipeline (RAGAS)")
    print("=" * 65)

    queries = get_custom_queries()
    naive_results = run_rag_on_queries(NaiveRAG(), queries, "Naive RAG")
    advanced_results = run_rag_on_queries(AdvancedRAG(), queries, "Advanced RAG")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ragas_llm = build_ragas_llm()
    ragas_emb = CustomOllamaEmbeddings()
    selected_metrics = [faithfulness, answer_relevancy]

    def get_scores(rag_results):
        data = {
            "question": [str(r["question"]) for r in rag_results],
            "answer":   [str(r["answer"])   for r in rag_results],
            "contexts": [list(r["contexts"]) if r["contexts"] else ["No context"] for r in rag_results],
        }
        return evaluate(Dataset.from_dict(data), metrics=selected_metrics, llm=ragas_llm, embeddings=ragas_emb)

    print("\n📊 Computing RAGAS scores...")
    n_res = get_scores(naive_results)
    a_res = get_scores(advanced_results)

    # ── Safe Results Extraction ──────────────────────────────────────────────

    def extract_metrics(result_obj):
        """Extracts averages safely from the EvaluationResult object."""
        scores = {}
        if result_obj is None: return scores
        
        try:
            scores = result_obj.to_pandas().mean(numeric_only=True).to_dict()
        except:
            for m in selected_metrics:
                try: scores[m.name] = result_obj[m.name]
                except: scores[m.name] = 0.0
        return scores

    clean_n = extract_metrics(n_res)
    clean_a = extract_metrics(a_res)

    print("\n" + "=" * 65)
    print("  RESULTS COMPARISON")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Naive RAG':>12} {'Advanced RAG':>14} {'Winner':>8}")
    print("  " + "-" * 60)
    
    for m in selected_metrics:
        name = m.name
        nv = clean_n.get(name, 0.0)
        av = clean_a.get(name, 0.0)
        winner = "Advanced" if av > nv else ("Naive" if nv > av else "Tie")
        print(f"  {name:<25} {nv:>12.4f} {av:>14.4f} {winner:>8}")
    print("=" * 65)

    # ── Export ───────────────────────────────────────────────────────────────
    summary = {"timestamp": timestamp, "naive_scores": clean_n, "advanced_scores": clean_a}
    with open(RESULTS_DIR / f"summary_{timestamp}.json", "w") as f: json.dump(summary, f, indent=4)
    
    rows = []
    for i, (nr, ar) in enumerate(zip(naive_results, advanced_results)):
        rows.append({
            "query_id": i + 1, "question": nr["question"],
            "naive_answer": nr["answer"], "advanced_answer": ar["answer"],
            "naive_ctx": len(nr["contexts"]), "adv_ctx": len(ar["contexts"])
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"answers_{timestamp}.csv", index=False)
    print(f"\n💾 Results saved to {RESULTS_DIR}")
    return summary

if __name__ == "__main__":
    evaluate_both()
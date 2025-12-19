import gradio as gr
import numpy as np
from typing import List, Tuple, Dict
import wikipedia
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sqlite3
import os
import json
import re

CACHE_PATH = "wikipedia_cache.json"
DB_PATH = "metadata.db"


def get_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def metadata_exists() -> bool:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
        count = cursor.fetchone()[0]
        return count > 0


def setup_database() -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                chunk_id TEXT PRIMARY KEY,
                source_title TEXT,
                url TEXT,
                document_type TEXT,
                topic TEXT,
                difficulty_level TEXT,
                year_mentioned INTEGER,
                categories TEXT,
                chunk_index INTEGER
            )
        ''')
        conn.commit()

def extract_all_years(text: str) -> int | None:
    years = re.findall(r'\b(1[5-9]\d{2}|20\d{2})\b', text)
    if not years:
        return None
    nums = [int(y) for y in years if 1500 <= int(y) <= 2100]
    if not nums:
        return None
    return int(np.median(nums))


def load_wikipedia_data(topics, lang="en") -> list:

    if os.path.exists(CACHE_PATH):
        print(f"Loaded Wikipedia data from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    wikipedia.set_lang(lang)
    documents = []

    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)

            year = extract_all_years(page.content)

            documents.append({
                "title": page.title,
                "url": page.url,
                "content": page.content,
                "categories": page.categories if hasattr(page, "categories") else [],
                "year_mentioned": year
            })

            print(f"Downloaded: {page.title}")

        except Exception as e:
            print(f"Error for {topic}: {e}")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Saved Wikipedia data to cache: {CACHE_PATH}")

    return documents


def create_chunks(documents: List[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc['content'])
        for i, chunk in enumerate(splits):
            chunk_id = f"{doc['title']}_chunk_{i}"
            chunks.append({
                'text': chunk,
                'chunk_id': chunk_id,
                'source': doc['title'],
                'url': doc['url'],
                'categories': doc.get('categories', []),
                'year_mentioned': doc.get('year_mentioned'),
                'chunk_index': i
            })

    return chunks


def extract_metadata_with_llm(chunk: dict, groq_client: Groq) -> dict:
    if not groq_client:
        return {'document_type': 'article', 'topic': 'general', 'difficulty_level': 'intermediate'}

    prompt = f"""Extract metadata as JSON ONLY.

Text:
{chunk['text'][:300]}

Return JSON with keys:
document_type, topic, difficulty_level
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120
        )
        result = response.choices[0].message.content.strip()
        result = result.replace('```json', '').replace('```', '')
        return json.loads(result)
    except:
        return {'document_type': 'article', 'topic': 'general', 'difficulty_level': 'intermediate'}


def score_categories_semantically(categories: List[str], text: str, embedder) -> List[str]:
    if not categories:
        return []

    cat_emb = embedder.encode(categories, convert_to_numpy=True)
    txt_emb = embedder.encode(text[:500], convert_to_numpy=True)

    sims = np.dot(cat_emb, txt_emb) / (
        np.linalg.norm(cat_emb, axis=1) * np.linalg.norm(txt_emb) + 1e-9
    )

    z = (sims - sims.mean()) / (sims.std() + 1e-9)
    keep = [cat for cat, s in zip(categories, z) if s > 0]

    return keep


def save_metadata_to_db(
    chunks: List[dict],
    groq_client: Groq = None,
    max_llm_chunks: int = 50
) -> None:

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with get_db() as conn:
        cursor = conn.cursor()
        print("Generating metadata...")

        for i, chunk in enumerate(chunks):

            if groq_client and i < max_llm_chunks:
                meta = extract_metadata_with_llm(chunk, groq_client)
            else:
                meta = {
                    "document_type": "article",
                    "topic": "general",
                    "difficulty_level": "intermediate"
                }

            document_type = meta.get("document_type", "article").lower()
            topic = meta.get("topic", "general").lower()
            difficulty = meta.get("difficulty_level", "intermediate").lower()

            year = chunk.get("year_mentioned")
            if isinstance(year, int) and not (1500 <= year <= 2100):
                year = None

            cleaned_categories = score_categories_semantically(
                chunk.get("categories", []),
                chunk["text"],
                embedder
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO chunk_metadata
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk["source"],
                    chunk["url"],
                    document_type,
                    topic,
                    difficulty,
                    year,
                    json.dumps(cleaned_categories, ensure_ascii=False),
                    chunk["chunk_index"]
                )
            )

        conn.commit()
        print("Metadata saved.")


def generate_metadata_filter(query: str, groq_client: Groq) -> Dict:
    if not groq_client:
        return {}

    prompt = f"""
Your task is to convert a natural-language query into structured metadata filters.

You MUST return a VALID JSON object with ANY of these optional fields:
- document_type: one of ["article","tutorial","technical","reference","academic"]
- topic: short keyword like "ai","machine learning","deep learning"
- difficulty_level: one of ["beginner","intermediate","advanced"]
- year_mentioned: an integer year

Rules:
1. NEVER return an empty JSON object unless query is completely unrelated to knowledge queries.
2. If the query contains words like:
   - "technical", "academic", "paper", "research" → document_type = "academic" or "technical"
   - "advanced", "expert" → difficulty_level = "advanced"
   - "beginner", "intro", "basics" → difficulty_level = "beginner"
3. If the query mentions a field (AI, ML, NLP, deep learning) set topic accordingly.
4. Respond ONLY with JSON. No text before or after.

Query: "{query}"
JSON:
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]

        return json.loads(json_text)

    except Exception as e:
        print("Filter parse error:", e)
        return {}



def apply_metadata_filter(
    candidate_indices: List[int],
    filters: Dict,
    chunks: List[dict],
    min_keep: int = 5
) -> List[int]:

    if not filters:
        return candidate_indices

    scored = []

    with get_db() as conn:
        cursor = conn.cursor()

        for idx in candidate_indices:
            chunk_id = chunks[idx]["chunk_id"]

            cursor.execute("""
                SELECT document_type, topic, difficulty_level, year_mentioned, categories
                FROM chunk_metadata
                WHERE chunk_id = ?
            """, (chunk_id,))
            row = cursor.fetchone()

            if not row:
                continue

            meta = {
                "document_type": row[0],
                "topic": row[1],
                "difficulty_level": row[2],
                "year_mentioned": row[3],
                "categories": json.loads(row[4]) if row[4] else []
            }

            score = 0

            for key, wanted in filters.items():
                if wanted is None:
                    continue

                actual = meta.get(key)
                if actual is None:
                    continue

                if key == "year_mentioned":
                    try:
                        wanted = int(wanted)
                        if actual == wanted:
                            score += 2
                        elif abs(actual - wanted) <= 5:
                            score += 1
                    except:
                        pass

                elif key == "topic":
                    if wanted.lower() == actual.lower():
                        score += 2
                    else:
                        score += 0.5 * (
                            wanted.lower() in actual.lower() or actual.lower() in wanted.lower()
                        )

                elif key == "document_type":
                    if wanted.lower() == actual.lower():
                        score += 2

                elif key == "difficulty_level":
                    if wanted.lower() == actual.lower():
                        score += 1.5

                elif key == "categories":
                    if isinstance(wanted, str):
                        wanted = [wanted]
                    overlap = len(set(wanted) & set(meta["categories"]))
                    score += overlap * 1.5

            scored.append((idx, score))

    if not scored:
        return candidate_indices[:min_keep]

    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [idx for idx, s in scored if s > 0]

    if len(filtered) >= min_keep:
        return filtered

    return [idx for idx, _ in scored[:min_keep]]


class RAGSystemWithMetadata:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self.texts = [c['text'] for c in chunks]
        self.client = None

        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = self.embedder.encode(self.texts, convert_to_numpy=True)
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def hybrid_search(self, query: str):
        bm = self.bm25.get_scores(query.lower().split())
        qemb = self.embedder.encode(query, convert_to_numpy=True)
        sims = np.dot(self.embeddings, qemb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(qemb) + 1e-9
        )
        return list(set(np.argsort(bm)[-25:]) | set(np.argsort(sims)[-25:]))

    def answer_question(self, query, use_bm25, use_dense, use_reranker, use_metadata, top_k):
        debug_info = "Searching:\n\n"

        candidates = self.hybrid_search(query)
        debug_info += f"Candidates - hybrid search: {len(candidates)}\n"
        debug_info += f"   - BM25: {'✅' if use_bm25 else '❌'}\n"
        debug_info += f"   - Dense: {'✅' if use_dense else '❌'}\n\n"

        if use_metadata and self.client:
            filters = generate_metadata_filter(query, self.client)
            if filters:
                debug_info += " Metadata Filtering: ✅\n"
                debug_info += f"   Filters: {json.dumps(filters, ensure_ascii=False)}\n"
                before = len(candidates)
                candidates = apply_metadata_filter(candidates, filters, self.chunks)
                debug_info += f"   Candidates after filters: {len(candidates)} (were {before})\n\n"
            else:
                debug_info += " Metadata Filtering: ⚠️ (no filters)\n\n"
        else:
            debug_info += " Metadata Filtering:  (off or some LLM error)\n\n"

        if use_reranker and candidates:
            pairs = [[query, self.texts[i]] for i in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
            debug_info += f"Reranker:  (top-{len(ranked)})\n\n"
        else:
            ranked = [(i, 0.0) for i in candidates[:top_k]]
            debug_info += "Reranker:  (skipped)\n\n"

        contexts = []
        with get_db() as conn:
            cursor = conn.cursor()
            for idx, score in ranked:
                chunk = self.chunks[idx]

                cursor.execute(
                    "SELECT document_type, topic, difficulty_level FROM chunk_metadata WHERE chunk_id = ?",
                    (chunk["chunk_id"],)
                )
                m = cursor.fetchone()

                contexts.append({
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "url": chunk["url"],
                    "score": score,
                    "metadata": {
                        "type": m[0] if m else "unknown",
                        "topic": m[1] if m else "unknown",
                        "difficulty": m[2] if m else "unknown"
                    }
                })

        debug_info += "Generate answer: "
        if not self.client:
            debug_info += " (LLM error)\n"
            return " Add Groq API key", contexts, debug_info

        context_text = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
        prompt = f"""Answer only from context. Use citations [1], [2].

    Context:
    {context_text}

    Question: {query}"""

        res = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800
        )

        debug_info += "✅\n"
        return res.choices[0].message.content, contexts, debug_info



setup_database()


topics = [
    # Core AI
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural networks",
    "Symbolic artificial intelligence",
    "Expert system",

    # NLP / LLM
    "Natural language processing",
    "Large language model",
    "Language model",
    "Transformer (machine learning model)",
    "Attention mechanism",
    "Document classification",
    "Text summarization",
    "Speech recognition",
    "Question answering",

    # Generative AI
    "Generative artificial intelligence",
    "Generative model",
    "Foundation model",
    "Diffusion model",
    "Generative adversarial network",
    "Variational autoencoder",
    "Prompt engineering",

    # Embeddings / Retrieval / RAG
    "Embedding (machine learning)",
    "Vector database",
    "Semantic search",
    "Similarity search",
    "Information retrieval",
    "Retrieval-augmented generation",
    "Question answering system",

    # Agentic / Reasoning
    "Agentic AI",
    "Intelligent agent",
    "Reinforcement learning",
    "Automated planning",
    "Multi-agent system",

    # Popular models / systems
    "ChatGPT",
    "GPT-4",
    "BERT (language model)",
    "OpenAI",

    # Training / Evaluation
    "Fine-tuning (deep learning)",
    "Transfer learning",
    "Explainable artificial intelligence",
    "AI alignment"
]
docs = load_wikipedia_data(topics)
chunks = create_chunks(docs)
rag_system = RAGSystemWithMetadata(chunks)


def answer_with_sources(query, api_key, use_bm25, use_dense, use_reranker, use_metadata, top_k):
    if api_key:
        rag_system.client = Groq(api_key=api_key)

        if not metadata_exists():
            print("Generating metadata...")
            save_metadata_to_db(chunks, rag_system.client)
    else:
        return (
            " Please enter your Groq API key",
            "",
            "No API key provided",
            get_available_metadata_filters()
        )
    answer, contexts, debug = rag_system.answer_question(
        query, use_bm25, use_dense, use_reranker, use_metadata, int(top_k)
    )

    sources = "\n\n **Sources (with metadata):**\n\n"
    for i, c in enumerate(contexts, 1):
        sources += (
            f"**[{i}] {c['source']}**\n"
            f"Score: {c['score']:.4f}\n"
            f"Type: {c['metadata']['type']} | Topic: {c['metadata']['topic']} | Difficulty: {c['metadata']['difficulty']}\n"
            f"URL: {c['url']}\n\n"
        )

    return answer, sources, debug, get_available_metadata_filters()


def get_available_metadata_filters() -> str:
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT document_type FROM chunk_metadata")
        types = [row[0] for row in cursor.fetchall() if row[0]]

        cursor.execute("SELECT DISTINCT topic FROM chunk_metadata")
        topics = [row[0] for row in cursor.fetchall() if row[0]]

        cursor.execute("SELECT DISTINCT difficulty_level FROM chunk_metadata")
        diffs = [row[0] for row in cursor.fetchall() if row[0]]


        md = "### Metadata Filtering \n"
        md += f"**Type:** {', '.join(types) if types else '—'}\n\n"
        md += f"**Topic:** {', '.join(topics) if topics else '—'}\n\n"
        md += f"**Difficulty:** {', '.join(diffs) if diffs else '—'}\n\n"

        return md

with gr.Blocks(title=" RAG QA System + Metadata", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Question Answering System + Metadata Filtering")
    metadata_md = gr.Markdown(
        "### Metadata Filtering\n\n_Generate metadata by asking a question_"
    )
    query_input = gr.Textbox(
        label=" Your question",
        placeholder="Find beginner tutorials about machine learning",
        lines=2
    )
    api_key_input = gr.Textbox(label="Groq API Key", type="password")

    with gr.Row():
        use_bm25 = gr.Checkbox(label=" BM25", value=True)
        use_dense = gr.Checkbox(label=" Dense", value=True)
        use_reranker = gr.Checkbox(label=" Reranker", value=True)
        use_metadata = gr.Checkbox(label=" Metadata Filtering", value=True)

    top_k = gr.Slider(1, 10, value=5, step=1, label="Num of sources")
    submit_btn = gr.Button(" Get an answer", variant="primary", size="lg")

    gr.Examples(
        examples=[
            ["What is machine learning?"],
            ["Find beginner tutorials about AI"],
            ["Show advanced technical articles about deep learning"],
            ["Explain neural networks for beginners"],
        ],
        inputs=query_input
    )

    answer_out = gr.Markdown()
    sources_out = gr.Markdown()
    with gr.Accordion(" Debug Info", open=False):
        debug_out = gr.Textbox(label="Searching", lines=10)

    submit_btn.click(
        answer_with_sources,
        [query_input, api_key_input, use_bm25, use_dense, use_reranker, use_metadata, top_k],
        [answer_out, sources_out, debug_out, metadata_md]
    )


    # gr.Markdown(get_available_metadata_filters())


demo.launch()

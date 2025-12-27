import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from app.config import settings
from app.rag import RAGService


def get_arxiv_root() -> Path:
    return Path(settings.RAW_DATA_DIR) / "official" / "pdf" / "arxiv"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def build_qrels_map(qrels_obj: Any) -> Dict[str, Tuple[str, int]]:
    """
    Returns: { query_id: (doc_id, section_id) }
    qrels formats vary, so we support list or dict.
    """
    out: Dict[str, Tuple[str, int]] = {}

    if isinstance(qrels_obj, dict):
        # Could be {qid: {...}} or {qid: [..]}
        for qid, rel in qrels_obj.items():
            if isinstance(rel, list) and rel:
                rel = rel[0]
            if isinstance(rel, dict):
                doc_id = str(pick_first(rel, ["doc_id", "id", "document_id", "corpus_id"]))
                section_id = int(pick_first(rel, ["section_id", "sec_id", "section", "passage_id"]) or 0)
                out[str(qid)] = (doc_id, section_id)
        return out

    if isinstance(qrels_obj, list):
        for rel in qrels_obj:
            if not isinstance(rel, dict):
                continue
            qid = pick_first(rel, ["query_id", "qid", "id", "uuid"])
            doc_id = pick_first(rel, ["doc_id", "document_id", "corpus_id", "doc"])
            section_id = pick_first(rel, ["section_id", "sec_id", "section", "passage_id"])
            if qid is None or doc_id is None or section_id is None:
                continue
            out[str(qid)] = (str(doc_id), int(section_id))
        return out

    return out


def main(num_eval: int = 100, k: int = 5):
    arxiv_root = get_arxiv_root()
    queries_path = arxiv_root / "queries.json"
    qrels_path = arxiv_root / "qrels.json"

    if not queries_path.exists() or not qrels_path.exists():
        raise FileNotFoundError(
            "queries.json/qrels.json not found. Run: python app/download_raw_dataset.py"
        )

    queries = load_json(queries_path)
    qrels_obj = load_json(qrels_path)
    qrels_map = build_qrels_map(qrels_obj)

    rag = RAGService()

    hits = 0
    total = 0

    for q in queries:
        if total >= num_eval:
            break
        if not isinstance(q, dict):
            continue

        qid = pick_first(q, ["query_id", "qid", "id", "uuid"])
        qtext = pick_first(q, ["text", "query", "question"])

        if qid is None or qtext is None:
            continue

        gold = qrels_map.get(str(qid))
        if not gold:
            continue

        gold_doc, gold_sec = gold

        retrieved = rag.retrieve(str(qtext), top_k=k)
        ok = any(
            str(d.metadata.get("doc_id", "")) == gold_doc
            and int(d.metadata.get("section_id", -1)) == int(gold_sec)
            for d in retrieved
        )

        hits += 1 if ok else 0
        total += 1

    print(f"Hit@{k}: {hits}/{total} = {hits / max(total, 1):.3f}")


if __name__ == "__main__":
    main(num_eval=100, k=5)

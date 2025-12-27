import json
from pathlib import Path
from typing import Any, Dict, List, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings


def get_arxiv_root() -> Path:
    """
    Handle both possible dataset layouts:
    1) data/open_ragbench_raw/official/pdf/arxiv
    2) data/open_ragbench_raw/pdf/arxiv
    """
    base = Path(settings.RAW_DATA_DIR)

    path1 = base / "official" / "pdf" / "arxiv"
    path2 = base / "pdf" / "arxiv"

    if path1.exists():
        return path1
    if path2.exists():
        return path2

    raise FileNotFoundError(
        "Could not find arxiv dataset directory.\n"
        f"Checked:\n  {path1}\n  {path2}"
    )


def section_to_text(section: Dict[str, Any]) -> str:
    text = (section.get("text") or "").strip()
    tables = section.get("tables") or {}

    if tables:
        table_blob = "\n\n".join([f"[TABLE]\n{t}" for t in tables.values()])
        text = f"{text}\n\n{table_blob}".strip()

    return text


def iter_paper_json_files(corpus_dir: Path, limit_docs: int | None) -> Iterable[Path]:
    files = sorted(corpus_dir.glob("*.json"))
    if limit_docs is not None:
        files = files[:limit_docs]
    for fp in files:
        yield fp


def load_documents(limit_docs: int | None = None) -> List[Document]:
    arxiv_root = get_arxiv_root()
    corpus_dir = arxiv_root / "corpus"

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus not found at: {corpus_dir}\n"
            f"Run: python app/download_raw_dataset.py"
        )

    docs: List[Document] = []
    for fp in iter_paper_json_files(corpus_dir, limit_docs):
        with fp.open("r", encoding="utf-8") as f:
            paper = json.load(f)

        doc_id = str(paper.get("id", fp.stem))
        title = paper.get("title", "") or ""
        sections = paper.get("sections") or []

        for section_id, sec in enumerate(sections):
            content = section_to_text(sec)
            if not content:
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "section_id": int(section_id),
                        "title": title,
                    },
                )
            )

    return docs


def build_chroma(limit_docs: int | None = None) -> None:
    raw_docs = load_documents(limit_docs=limit_docs)
    if not raw_docs:
        raise ValueError("No documents loaded from corpus.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DIR,
        collection_name=settings.CHROMA_COLLECTION,
    )
    vectordb.persist()

    print("Ingestion complete.")
    print("Papers processed:", limit_docs if limit_docs is not None else "ALL")
    print("Sections loaded:", len(raw_docs))
    print("Chunks indexed:", len(chunks))
    print("Chroma dir:", settings.CHROMA_DIR)


if __name__ == "__main__":
    build_chroma(limit_docs=settings.DEFAULT_DOC_LIMIT)

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Local data directory
    RAW_DATA_DIR: str = "data/open_ragbench_raw"

    # Local Chroma DB
    CHROMA_DIR: str = "db/chroma_open_ragbench"
    CHROMA_COLLECTION: str = "open_ragbench_arxiv"

    OLLAMA_HOST: str = "http://localhost:11434"  # default for non-Docker runs
    OLLAMA_MODEL: str = "llama3.2:3b"

    # Embedding model
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion knobs
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 150

    # Default limit for quick local runs
    DEFAULT_DOC_LIMIT: int = 100

    class Config:
        env_file = ".env"

settings = Settings()

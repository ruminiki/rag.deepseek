# config.py
from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    device: str = "cuda"  # Will fall back to CPU if CUDA not available

@dataclass
class RAGConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 1024
    memory_window: int = 4
    top_k_results: int = 5
    mmr_lambda: float = 0.5
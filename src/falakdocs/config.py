import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_dir: Path = field(default_factory=lambda: Path("index"))
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    provider: str = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_S"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    top_k: int = 5
    min_similarity: float = 0.45
    api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv
        return cls(
            index_dir=Path(env("FALAKDOCS_INDEX", "index")),
            ollama_url=env("OLLAMA_URL", cls.ollama_url),
            embed_model=env("FALAKDOCS_EMBED_MODEL", cls.embed_model),
            provider=env("FALAKDOCS_PROVIDER", cls.provider),
            ollama_model=env("FALAKDOCS_OLLAMA_MODEL", cls.ollama_model),
            claude_model=env("FALAKDOCS_CLAUDE_MODEL", cls.claude_model),
            claude_effort=env("FALAKDOCS_CLAUDE_EFFORT", cls.claude_effort),
            top_k=int(env("FALAKDOCS_TOP_K", "5")),
            min_similarity=float(env("FALAKDOCS_MIN_SIMILARITY", "0.45")),
            api_key=env("FALAKDOCS_API_KEY", ""),
        )

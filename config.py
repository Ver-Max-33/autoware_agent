from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

DEFAULT_COMPONENTS: Dict[str, List[str]] = {
    "architecture": [
        "https://autowarefoundation.github.io/autoware-documentation/main/design/autoware-architecture/index.html",
    ],
    "planning": [
        "https://autowarefoundation.github.io/autoware-documentation/main/design/autoware-architecture/planning.html",
    ],
    "perception": [
        "https://autowarefoundation.github.io/autoware-documentation/main/design/autoware-architecture/perception.html",
    ],
}


@dataclass
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    chroma_dir: Path = field(init=False)
    chat_model: str = field(
        default_factory=lambda: os.environ.get("AUTOWARE_AGENT_CHAT_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "AUTOWARE_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    openai_api_key: str = field(init=False)
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 4
    max_iterations: int = 3
    request_timeout: int = 60
    max_retries: int = 2
    components: Dict[str, List[str]] = field(
        default_factory=lambda: {name: list(urls) for name, urls in DEFAULT_COMPONENTS.items()}
    )

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir = self.data_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is required.")
        self.openai_api_key = key


def load_config() -> AppConfig:
    return AppConfig()

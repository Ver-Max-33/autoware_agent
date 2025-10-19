from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import requests
from langchain_core.documents import Document

from config import AppConfig

LOG = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_documents(self, components: Iterable[str] | None = None) -> List[Document]:
        targets = self._select_components(components)
        documents: List[Document] = []
        for name, urls in targets.items():
            for url in urls:
                document = self._fetch(url, name)
                if document:
                    documents.append(document)
        return documents

    def _select_components(self, requested: Iterable[str] | None) -> Dict[str, List[str]]:
        if requested is None:
            return self.config.components
        selected: Dict[str, List[str]] = {}
        missing: List[str] = []
        for name in requested:
            if name in self.config.components:
                selected[name] = self.config.components[name]
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Unknown components: {', '.join(missing)}")
        return selected

    def _fetch(self, url: str, component: str) -> Document | None:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOG.warning("Failed to fetch %s (%s)", url, exc)
            return None
        return Document(
            page_content=response.text,
            metadata={
                "source_url": url,
                "component": component,
            },
        )

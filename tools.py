from __future__ import annotations

import json
from typing import Any, List, Optional

from langchain_core.tools import tool
from langchain_core.documents import Document

from config import AppConfig
from vector_store import VectorStoreManager


class ToolManager:
    def __init__(self, config: AppConfig, vector_store: VectorStoreManager) -> None:
        self.config = config
        self.vector_store = vector_store

    def get_tools(self) -> List[Any]:
        @tool("search_documents")
        def search_documents_tool(query: str, k: Optional[int] = None) -> str:
            """Return top-k similar document snippets for the given query."""
            results = self.vector_store.similarity_search(
                query=query,
                k=k or self.config.top_k,
            )
            payload = [self._serialize_result(doc, index) for index, doc in enumerate(results, start=1)]
            return json.dumps(payload, ensure_ascii=False, indent=2)

        @tool("read_full_document")
        def read_full_document_tool(doc_id: str) -> str:
            """Return full cached chunks associated with a document id."""
            chunks = self.vector_store.fetch_document(doc_id)
            payload = [
                {
                    "index": chunk.metadata.get("chunk_index"),
                    "source_url": chunk.metadata.get("source_url"),
                    "component": chunk.metadata.get("component"),
                    "content": chunk.page_content,
                }
                for chunk in chunks
            ]
            return json.dumps(payload, ensure_ascii=False, indent=2)

        @tool("list_available_components")
        def list_components_tool() -> str:
            """Return the component names currently known to the agent."""
            components = self.vector_store.list_components()
            if not components:
                components = sorted(self.config.components.keys())
            return json.dumps({"components": components}, ensure_ascii=False, indent=2)

        return [
            search_documents_tool,
            read_full_document_tool,
            list_components_tool,
        ]

    @staticmethod
    def _serialize_result(doc: Document, score: int) -> dict[str, Any]:
        metadata = doc.metadata or {}
        return {
            "rank": score,
            "doc_id": metadata.get("doc_id"),
            "component": metadata.get("component"),
            "source_url": metadata.get("source_url"),
            "excerpt": doc.page_content[:400],
        }

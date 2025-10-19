from __future__ import annotations

import os
import hashlib
import logging
from typing import Dict, Iterable, List

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig

os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

LOG = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embedding = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.vector_store: Chroma | None = None

    def build(self, documents: Iterable[Document]) -> None:
        chunks: List[Document] = []
        for doc in documents:
            metadata = self._enrich_metadata(doc.metadata)
            doc_id = metadata["doc_id"]
            for index, chunk in enumerate(self.splitter.split_documents([doc])):
                chunk.metadata.update(metadata)
                chunk.metadata["chunk_index"] = index
                chunk.metadata["doc_id"] = doc_id
                chunks.append(chunk)
        if not chunks:
            raise ValueError("No documents available to index.")
        LOG.info("Indexing %d chunks", len(chunks))
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            collection_name="autoware_agent",
            persist_directory=str(self.config.chroma_dir),
            client_settings=Settings(anonymized_telemetry=False),
        )

    def similarity_search(self, query: str, k: int) -> List[Document]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not built.")
        return self.vector_store.similarity_search(query=query, k=k)

    def fetch_document(self, doc_id: str) -> List[Document]:
        if self.vector_store is None:
            raise RuntimeError("Vector store not built.")
        raw = self.vector_store.get(where={"doc_id": doc_id})
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        results: List[Document] = []
        for content, metadata in zip(documents, metadatas):
            results.append(Document(page_content=content, metadata=metadata or {}))
        results.sort(key=lambda item: item.metadata.get("chunk_index", 0))
        return results

    def list_components(self) -> List[str]:
        if self.vector_store is None:
            return []
        raw = self.vector_store.get(include=["metadatas"])
        components = {meta.get("component") for meta in raw.get("metadatas", [])}
        return sorted(filter(None, components))

    def _enrich_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        source = metadata.get("source_url", "unknown")
        component = metadata.get("component", "unknown")
        doc_id = metadata.get("doc_id") or hashlib.sha1(source.encode("utf-8")).hexdigest()
        return {
            "source_url": source,
            "component": component,
            "doc_id": doc_id,
        }

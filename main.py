from __future__ import annotations
import argparse
import logging
import sys
from typing import Iterable

from agent import AgentRunner
from config import load_config
from document_loader import DocumentLoader
from tools import ToolManager
from vector_store import VectorStoreManager


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoware agentic RAG demo.")
    parser.add_argument("--components", nargs="*", help="Limit document refresh to specific components.")
    parser.add_argument("--question", type=str, help="Ask a single question and exit.")
    parser.add_argument("--trace", action="store_true", help="Stream agent intermediate steps.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or [])
    configure_logging(args.verbose)
    logging.info("Starting Autoware documentation assistant")

    config = load_config()
    try:
        documents = DocumentLoader(config).load_documents(args.components)
    except ValueError as exc:
        logging.error(str(exc))
        return 1
    vector_store = VectorStoreManager(config)
    vector_store.build(documents)

    tools = ToolManager(config, vector_store).get_tools()
    agent = AgentRunner(config, tools)

    question = args.question or "Autoware の Planning コンポーネントについて教えてください"
    run_question(agent, question, trace=args.trace)
    return 0


def run_question(agent: AgentRunner, question: str, trace: bool) -> None:
    print("============================================")
    print(f"質問: {question}")
    if trace:
        print("---- Agent intermediate steps ----")
        for event in agent.stream(question):
            for node_name, value in event.items():
                print(f"[{node_name}] {value}")
        print("---- End of intermediate steps ----")
    answer = agent.invoke(question)
    print("---- 回答 ----")
    print(answer)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

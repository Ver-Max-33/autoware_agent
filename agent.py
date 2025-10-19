from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, TypedDict

from operator import add

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from config import AppConfig

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "あなたはAutowareの専門家として日本語で回答します。\n"
    "利用できるツールは次のとおりです:\n"
    "- search_documents: 関連するスニペットを検索\n"
    "- read_full_document: 指定ドキュメントの全文を確認\n"
    "- list_available_components: コンポーネント一覧を取得\n"
    "まず search_documents で情報を集め、不足があれば read_full_document を呼び出してください。\n"
    "回答は取得した情報のみに基づき、最後に参照した source_url を列挙してください。"
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: Annotated[int, add]


class AgentRunner:
    def __init__(
        self,
        config: AppConfig,
        tools: Iterable[Any],
    ) -> None:
        self.config = config
        self.tools = list(tools)
        self.max_iterations = config.max_iterations
        self.chat_model = ChatOpenAI(
            model=config.chat_model,
            temperature=0,
            max_retries=config.max_retries,
            timeout=config.request_timeout,
            api_key=config.openai_api_key,
        ).bind_tools(self.tools)
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()

    def _build_graph(self):
        max_iterations = self.max_iterations

        def call_model(state: AgentState) -> Dict[str, List[AIMessage]]:
            LOG.debug("Model node invoked with %d messages", len(state["messages"]))
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            if state.get("iterations", 0) >= max_iterations:
                messages.append(
                    SystemMessage(
                        content=(
                            "ツール呼び出し上限に達しました。既存の情報だけで簡潔に回答し、不足があればその旨を伝えてください。"
                        )
                    )
                )
            messages.extend(list(state["messages"]))
            response = self.chat_model.invoke(messages)
            return {"messages": [response]}

        def execute_tools(state: AgentState) -> Dict[str, List[ToolMessage]]:
            last_message = state["messages"][-1]
            tool_responses = []
            for tool_call in getattr(last_message, "tool_calls", []):
                tool_name = getattr(tool_call, "name", None) or getattr(
                    getattr(tool_call, "function", None), "name", None
                )
                if tool_name is None and isinstance(tool_call, dict):
                    tool_name = tool_call.get("name")
                tool = self.tool_map.get(tool_name)
                if tool is None:
                    LOG.warning("Tool %s requested but not registered.", tool_name)
                    continue
                args = self._parse_args(tool_call)
                LOG.info("Executing tool %s with args %s", tool_name, args)
                observation = tool.invoke(args or {})
                tool_responses.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=getattr(tool_call, "id", None)
                        or getattr(getattr(tool_call, "function", None), "id", None)
                        or (tool_call.get("id") if isinstance(tool_call, dict) else None),
                    )
                )
            return {"messages": tool_responses, "iterations": len(tool_responses)}

        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                if state.get("iterations", 0) >= max_iterations:
                    LOG.info("Tool iteration limit reached; stopping further tool usage.")
                    return END
                return "tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("model", call_model)
        graph.add_node("tools", execute_tools)
        graph.set_entry_point("model")
        graph.add_conditional_edges("model", should_continue)
        graph.add_edge("tools", "model")
        return graph.compile()

    def invoke(self, user_message: str, history: Optional[List[HumanMessage | AIMessage]] = None) -> str:
        history = history or []
        state_messages = history + [HumanMessage(content=user_message)]
        result = self.graph.invoke({"messages": state_messages, "iterations": 0})
        final_message = result["messages"][-1]
        return final_message.content

    def stream(self, user_message: str, history: Optional[List[Any]] = None):
        history = history or []
        state_messages = history + [HumanMessage(content=user_message)]
        for update in self.graph.stream({"messages": state_messages, "iterations": 0}):
            yield update

    @staticmethod
    def _parse_args(tool_call: Any) -> Dict[str, Any]:
        if hasattr(tool_call, "args"):
            args = getattr(tool_call, "args")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    return {}
            if isinstance(args, dict):
                return args
        if hasattr(tool_call, "function"):
            function_payload = getattr(tool_call, "function")
            raw_arguments = getattr(function_payload, "arguments", None)
            if isinstance(raw_arguments, str):
                try:
                    return json.loads(raw_arguments)
                except json.JSONDecodeError:
                    return {}
            if isinstance(raw_arguments, dict):
                return raw_arguments
        if isinstance(tool_call, dict):
            args = tool_call.get("args") or tool_call.get("arguments")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    return {}
            if isinstance(args, dict):
                return args
        return {}

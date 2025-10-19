from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List

TOOL_NAMES = [
    "search_documents",
    "read_full_document",
    "list_available_components",
]


def load_tool_names() -> List[str]:
    return list(TOOL_NAMES)


def build_mermaid(tool_names: Iterable[str]) -> str:
    tool_nodes = "\n        ".join(f"{name}[/{name}/]" for name in tool_names)
    tool_links = "\n        ".join(f"tools --> {name}" for name in tool_names)
    return textwrap.dedent(
        f"""
        flowchart LR
            start((User Prompt)) --> model[LangGraph Agent]
            model --> tools[Tools]
            tools --> model
            model --> answer((Answer))
            note{{Iteration limit}} -.-> answer
            subgraph Toolset
                direction TB
                {tool_nodes}
            end
            {tool_links}
        """
    ).strip()


def build_text(tool_names: Iterable[str]) -> str:
    lines = [
        "Agentic RAG Flow:",
        "  1. User prompt enters the LangGraph agent.",
        "  2. Agent calls tools as needed until it reaches the iteration limit.",
        "  3. Final answer is generated in Japanese with cited sources.",
        "",
        "Tools:",
    ]
    for name in tool_names:
        lines.append(f"  - {name}")
    return "\n".join(lines)


def build_svg(tool_names: Iterable[str]) -> str:
    tool_tspans: List[str] = []
    for index, name in enumerate(tool_names):
        dy = "0" if index == 0 else "18"
        tool_tspans.append(f'<tspan x="580" dy="{dy}">• {name}</tspan>')
    tool_text_block = "\n      ".join(tool_tspans) if tool_tspans else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 320" width="900" height="320">
  <defs>
    <style>
      * {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
      .title {{ font-size: 18px; font-weight: 700; fill: #0f172a; }}
      .subtitle {{ font-size: 14px; fill: #1f2937; }}
      .label {{ font-size: 13px; fill: #475569; }}
      .bubble {{ fill: #f0f9ff; stroke: #38bdf8; stroke-width: 1.2; rx: 20; ry: 20; }}
      .panel {{ fill: #f8fafc; stroke: #cbd5f5; stroke-width: 1.2; rx: 18; ry: 18; }}
      .circle {{ fill: #f1f5f9; stroke: #94a3b8; stroke-width: 1.2; }}
      .chip {{ fill: #e0f2fe; stroke: #0ea5e9; stroke-width: 1; rx: 12; ry: 12; }}
      .chip-text {{ font-size: 12px; fill: #0c4a6e; font-weight: 600; }}
      .note {{ font-size: 12px; fill: #0f172a; }}
    </style>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,8 L10,4 z" fill="#0f172a" />
    </marker>
    <marker id="arrowGray" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,8 L10,4 z" fill="#64748b" />
    </marker>
  </defs>

  <rect x="0" y="0" width="900" height="320" fill="#ffffff" rx="28" ry="28"/>

  <circle class="circle" cx="110" cy="160" r="46"/>
  <text class="title" x="110" y="154" text-anchor="middle">User</text>
  <text class="subtitle" x="110" y="178" text-anchor="middle">Prompt</text>

  <rect class="bubble" x="180" y="80" width="260" height="160"/>
  <text class="title" x="310" y="115" text-anchor="middle">LangGraph Agent</text>
  <text class="subtitle" x="310" y="140" text-anchor="middle">Routes between reasoning and tools</text>
  <rect class="chip" x="240" y="160" width="140" height="28"/>
  <text class="chip-text" x="310" y="178" text-anchor="middle">SYSTEM PROMPT</text>
  <text class="label" x="310" y="204" text-anchor="middle">Replies in Japanese with sources.</text>

  <rect class="panel" x="470" y="80" width="220" height="160"/>
  <text class="title" x="580" y="115" text-anchor="middle">Tools</text>
  <text class="subtitle" x="580" y="140" text-anchor="middle">Agentic RAG actions</text>
  <text class="label" x="580" y="168" text-anchor="middle">
      {tool_text_block}
  </text>

  <circle class="circle" cx="780" cy="160" r="46"/>
  <text class="title" x="780" y="154" text-anchor="middle">Final</text>
  <text class="subtitle" x="780" y="178" text-anchor="middle">Answer</text>

  <line x1="156" y1="160" x2="180" y2="160" stroke="#0f172a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="440" y1="160" x2="470" y2="160" stroke="#0f172a" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="690" y1="160" x2="734" y2="160" stroke="#0f172a" stroke-width="2" marker-end="url(#arrow)"/>

  <path d="M580 240 C 450 300 370 260 330 220" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrowGray)" />

  <rect class="panel" x="350" y="250" width="200" height="50" rx="12" ry="12"/>
  <text class="note" x="450" y="280" text-anchor="middle">Iteration limit: 3 tool rounds</text>
</svg>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the Autoware documentation agent topology."
    )
    parser.add_argument(
        "--format",
        choices=("mermaid", "text", "svg"),
        default="mermaid",
        help="Output format for the visualization (default: mermaid).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional file path to save the visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools = load_tool_names()
    if args.format == "mermaid":
        content = build_mermaid(tools)
        if args.output:
            Path(args.output).write_text(content, encoding="utf-8")
            print(f"Visualization written to {args.output}")
        else:
            print(content)
        return
    if args.format == "text":
        content = build_text(tools)
        if args.output:
            Path(args.output).write_text(content, encoding="utf-8")
            print(f"Visualization written to {args.output}")
        else:
            print(content)
        return
    content = build_svg(tools)
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Visualization written to {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()

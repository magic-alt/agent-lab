from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p00 = sub.add_parser("phase00", help="Responses API quickstart")
    p00.add_argument("prompt")

    sub.add_parser("phase01", help="offline Transformer attention experiment")

    p02 = sub.add_parser("phase02", help="offline token/sampling experiment")
    p02.add_argument("--text", default="Transformer, RAG, MCP and Agent")

    sub.add_parser("phase03", help="offline context packing experiment")

    p04 = sub.add_parser("phase04", help="online strict tool-calling loop")
    p04.add_argument("prompt")
    p04.add_argument("--file-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phase00":
        from agent_lab.experiments.phase00_quickstart import run

        print(run(args.prompt))
    elif args.command == "phase01":
        from agent_lab.experiments.phase01_transformer import run

        run()
    elif args.command == "phase02":
        from agent_lab.experiments.phase02_tokens_sampling import run

        run(args.text)
    elif args.command == "phase03":
        from agent_lab.experiments.phase03_context import run

        run()
    elif args.command == "phase04":
        from agent_lab.experiments.phase04_tools import run

        print(run(args.prompt, file_root=args.file_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

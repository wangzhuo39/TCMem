from __future__ import annotations

import argparse
import json
from typing import Any

from ..serialization import to_primitive
from . import realmem_top_session
from .realmem_top_session import EvaluationOptions


def evaluation_options_from_args(_args: argparse.Namespace) -> EvaluationOptions:
    return EvaluationOptions(mode="tcmem_no_task_chain_online", task_chain_enabled=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return realmem_top_session.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    return realmem_top_session.run_evaluation(args, options=evaluation_options_from_args(args))


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

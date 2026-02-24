from __future__ import annotations

import argparse

from experiments.run_prob import build_parser as build_prob_parser
from experiments.run_prob import run as run_prob


def build_parser() -> argparse.ArgumentParser:
    parser = build_prob_parser()
    parser.description = (
        "Compatibility wrapper: delegates to experiments.run_prob so coverage/PIT "
        "always uses the shared probabilistic pipeline."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_prob(args)


if __name__ == "__main__":
    main()

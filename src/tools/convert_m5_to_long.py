from __future__ import annotations

import argparse
from pathlib import Path

from data_loading import convert_m5_wide_to_long


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert M5 wide format to long format")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sales_train_evaluation.csv"),
        help="Path to M5 wide format file (e.g., sales_train_evaluation.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/m5_evaluation_long.csv"),
        help="Path to save long format output",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(
            "Input file not found. Download the M5 dataset from "
            "https://www.kaggle.com/c/m5-forecasting-accuracy/data"
        )

    print(f"Converting {args.input} to long format...")
    print("This may take a few minutes for large files...")

    df_long = convert_m5_wide_to_long(args.input, output_path=args.output)

    print("\n✓ Conversion complete!")
    print(f"  Output shape: {df_long.shape}")
    print(f"  Saved to: {args.output}")
    print(f"  File size: {args.output.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()

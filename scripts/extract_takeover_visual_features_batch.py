#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.extract_takeover_visual_features import DEFAULT_FEATURE_KEY
from scripts.extract_takeover_visual_features import extract_features_for_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-aligned pi0.5 feature extraction for multiple configs. "
            "Each --run item has the form CONFIG_NAME:TARGETS_PARQUET:OUTPUT_DIR."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Extraction spec: CONFIG_NAME:TARGETS_PARQUET:OUTPUT_DIR. May be repeated.",
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--feature-key", default=DEFAULT_FEATURE_KEY)
    return parser.parse_args()


def _parse_run_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --run spec {spec!r}; expected CONFIG_NAME:TARGETS_PARQUET:OUTPUT_DIR.")
    config_name, targets, output_dir = parts
    return config_name, Path(targets), Path(output_dir)


def main() -> None:
    args = parse_args()
    for spec in args.run:
        config_name, targets, output_dir = _parse_run_spec(spec)
        extract_features_for_config(
            config_name=config_name,
            targets=targets,
            output_dir=output_dir,
            batch_size=args.batch_size,
            feature_key=args.feature_key,
        )


if __name__ == "__main__":
    main()

"""Command-line entry point."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "prepare-projects": "qdmouse4m_benchmark.prepare_projects",
    "train-lp3d": "qdmouse4m_benchmark.train_lp3d",
    "infer-lp3d": "qdmouse4m_benchmark.infer_lp3d",
    "train-sleap": "qdmouse4m_benchmark.train_sleap",
    "infer-sleap": "qdmouse4m_benchmark.infer_sleap",
    "evaluate": "qdmouse4m_benchmark.evaluate",
    "materialize-images": "qdmouse4m_benchmark.materialize_images",
    "convert-sleap": "qdmouse4m_benchmark.convert_sleap_predictions",
    "kpms": "qdmouse4m_benchmark.kpms",
    "gait": "qdmouse4m_benchmark.gait",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = "\n  ".join(sorted(COMMANDS))
        print(f"usage: qdmouse4m <command> [args]\n\ncommands:\n  {commands}")
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        commands = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"Unknown command: {command}\nAvailable commands: {commands}")
    sys.argv = [f"qdmouse4m {command}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()

"""Single production entry point for the fare-prediction pipeline.

Every operational stage is a subcommand here instead of a separate
`python -m src.<module>` invocation to remember. Each subcommand dispatches
to that stage's existing standalone module unchanged -- this is a thin
router, not a rewrite -- so every module's own --help and arguments work
exactly as before, just reached through one command.

Run:
    python -m src.cli train        --sample 5000 --tag ci-abc123
    python -m src.cli evaluate     --tag ci-abc123
    python -m src.cli sweep        --model all --tag hp-metadata
    python -m src.cli baseline     --sample 5000 --tag baseline
    python -m src.cli analyze
    python -m src.cli mitigate     --drift-parquet training_set/yellow_tripdata_2024-09.parquet
    python -m src.cli mitigate-walkforward
    python -m src.cli explain-fare --tag metadata-final

    python -m src.cli <command> --help   # forwards to that stage's own argparse
"""

import importlib
import sys

COMMANDS = {
    "train": "src.train",
    "evaluate": "src.evaluate",
    "sweep": "src.sweep",
    "baseline": "src.baseline",
    "analyze": "src.analyze",
    "mitigate": "src.mitigate",
    "mitigate-walkforward": "src.mitigate_walkforward",
    "explain-fare": "src.fare_breakdown",
}


def main():
    if (
        len(sys.argv) < 2
        or sys.argv[1] in ("-h", "--help")
        or sys.argv[1] not in COMMANDS
    ):
        print("Usage: python -m src.cli <command> [args...]\n")
        print("Commands:")
        for name in COMMANDS:
            print(f"  {name}")
        print(
            "\nRun `python -m src.cli <command> --help` for a command's own arguments."
        )
        return 0 if len(sys.argv) < 2 else 1

    command = sys.argv[1]
    module_name = COMMANDS[command]
    module = importlib.import_module(module_name)

    # Forward the remaining args to the target module's own argparse, as if
    # it had been invoked directly (python -m <module_name> [args...]).
    sys.argv = [module_name] + sys.argv[2:]
    return module.main()


if __name__ == "__main__":
    sys.exit(main())

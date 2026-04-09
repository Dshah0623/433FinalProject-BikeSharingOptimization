from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "interim" / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


STEPS = [
    "01_validate_data",
    "02_eda",
    "03_build_features",
    "04_train_models",
    "05_generate_forecasts",
    "06_capacity_policy",
    "07_zone_rebalancing",
    "08_scenarios",
]


def main() -> None:
    for module_name in STEPS:
        print(f"Running {module_name}...")
        module = importlib.import_module(module_name)
        module.main()
    print("Pipeline completed.")


if __name__ == "__main__":
    main()

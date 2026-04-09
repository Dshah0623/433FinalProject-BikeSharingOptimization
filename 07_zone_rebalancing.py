from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bikeshare_project.config import load_config
from bikeshare_project.io_utils import save_dataframe
from bikeshare_project.optimization import derive_zone_targets, run_zone_sensitivity, solve_zone_rebalancing
from bikeshare_project.paths import RESULTS_RECOMMENDATIONS


def main() -> None:
    config = load_config()
    availability_plan = pd.read_csv(RESULTS_RECOMMENDATIONS / "availability_plan.csv", parse_dates=["timestamp", "date"])
    zone_targets = derive_zone_targets(availability_plan, config)
    current_inventory = {zone: int(value) for zone, value in config["optimization"]["default_current_inventory"].items()}
    shortage_cost = float(config["optimization"]["shortage_cost"])
    idle_cost = float(config["optimization"]["idle_cost"])
    move_cost = float(config["optimization"]["move_cost"])
    move_capacity = int(config["optimization"]["move_capacity"])

    zone_summary, move_plan = solve_zone_rebalancing(
        zone_targets=zone_targets,
        current_inventory=current_inventory,
        move_cost=move_cost,
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        move_capacity=move_capacity,
    )
    sensitivity = run_zone_sensitivity(
        zone_targets=zone_targets,
        current_inventory=current_inventory,
        move_cost_values=[move_cost * 0.5, move_cost, move_cost * 1.5, move_cost * 2.0],
        shortage_cost=shortage_cost,
        idle_cost=idle_cost,
        move_capacity=move_capacity,
    )

    save_dataframe(zone_targets, RESULTS_RECOMMENDATIONS / "zone_targets.csv")
    save_dataframe(move_plan, RESULTS_RECOMMENDATIONS / "zone_move_plan.csv")
    save_dataframe(sensitivity, RESULTS_RECOMMENDATIONS / "zone_sensitivity.csv")
    save_dataframe(zone_summary, RESULTS_RECOMMENDATIONS / "zone_plan_summary.csv")
    print("Saved zone target, move plan, and move-cost sensitivity outputs.")


if __name__ == "__main__":
    main()

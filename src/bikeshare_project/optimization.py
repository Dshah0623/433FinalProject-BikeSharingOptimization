from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import pulp


def critical_fractile(shortage_cost: float, idle_cost: float) -> float:
    return shortage_cost / (shortage_cost + idle_cost)


def _residual_adjustment(residuals: Iterable[float], quantile: float) -> float:
    residuals_array = np.asarray(list(residuals), dtype=float)
    if residuals_array.size == 0:
        return 0.0
    return float(np.quantile(residuals_array, quantile))


def build_availability_plan(
    forecast_df: pd.DataFrame,
    residuals: Iterable[float],
    shortage_cost: float,
    idle_cost: float,
    fleet_size: float,
) -> pd.DataFrame:
    tau = critical_fractile(shortage_cost, idle_cost)
    adjustment = _residual_adjustment(residuals, tau)
    plan = forecast_df.copy()
    plan["critical_fractile"] = tau
    plan["quantile_adjustment"] = adjustment
    plan["planned_availability"] = (plan["prediction"] + adjustment).clip(lower=0, upper=fleet_size)
    if "cnt" in plan.columns:
        plan["proxy_shortage"] = (plan["cnt"] - plan["planned_availability"]).clip(lower=0)
        plan["proxy_idle"] = (plan["planned_availability"] - plan["cnt"]).clip(lower=0)
        plan["weighted_cost_index"] = shortage_cost * plan["proxy_shortage"] + idle_cost * plan["proxy_idle"]
        plan["service_level_proxy"] = np.where(plan["cnt"] > 0, np.minimum(plan["planned_availability"], plan["cnt"]) / plan["cnt"], 1.0)
        plan["utilization_proxy"] = np.where(plan["planned_availability"] > 0, np.minimum(plan["cnt"], plan["planned_availability"]) / plan["planned_availability"], 0.0)
    return plan


def summarize_plan_vs_baseline(
    availability_plan: pd.DataFrame,
    baseline_plan: pd.DataFrame,
    shortage_cost: float,
    idle_cost: float,
) -> pd.DataFrame:
    def summarize(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        proxy_shortage = df.get("proxy_shortage", pd.Series(dtype=float)).sum()
        proxy_idle = df.get("proxy_idle", pd.Series(dtype=float)).sum()
        weighted_cost = shortage_cost * proxy_shortage + idle_cost * proxy_idle
        return {
            "policy": label,
            "mean_planned_availability": df["planned_availability"].mean(),
            "total_proxy_shortage": proxy_shortage,
            "total_proxy_idle": proxy_idle,
            "weighted_cost_index": weighted_cost,
            "mean_service_level_proxy": df.get("service_level_proxy", pd.Series(dtype=float)).mean(),
            "mean_utilization_proxy": df.get("utilization_proxy", pd.Series(dtype=float)).mean(),
        }

    return pd.DataFrame(
        [
            summarize(availability_plan, "forecast_driven"),
            summarize(baseline_plan, "static_hourly_mean"),
        ]
    )


def derive_zone_targets(
    availability_plan: pd.DataFrame,
    config: Dict[str, Any],
    zone_shares: Dict[str, float] | None = None,
) -> pd.DataFrame:
    workingday_flag = int(availability_plan["workingday"].mode().iloc[0])
    key = "workingday" if workingday_flag == 1 else "non_workingday"
    service_hours = config["optimization"]["default_service_hours"][key]
    active = availability_plan.loc[availability_plan["hr"].isin(service_hours)].copy()
    total_target = float(active["planned_availability"].max())
    shares = zone_shares or config["optimization"]["default_zone_shares"][key]
    rows = []
    for zone_name, share in shares.items():
        rows.append({"zone": zone_name, "share": share, "target_inventory": total_target * share})
    zone_targets = pd.DataFrame(rows)
    zone_targets["target_inventory"] = zone_targets["target_inventory"].round(0).astype(int)
    return zone_targets


def solve_zone_rebalancing(
    zone_targets: pd.DataFrame,
    current_inventory: Dict[str, int],
    move_cost: float,
    shortage_cost: float,
    idle_cost: float,
    move_capacity: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zones = zone_targets["zone"].tolist()
    target_lookup = zone_targets.set_index("zone")["target_inventory"].to_dict()
    model = pulp.LpProblem("overnight_zone_rebalancing", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer")
        for i in zones
        for j in zones
        if i != j
    }
    shortage = {i: pulp.LpVariable(f"shortage_{i}", lowBound=0, cat="Integer") for i in zones}
    excess = {i: pulp.LpVariable(f"excess_{i}", lowBound=0, cat="Integer") for i in zones}

    model += (
        pulp.lpSum(move_cost * x_var for x_var in x.values())
        + pulp.lpSum(shortage_cost * shortage_var for shortage_var in shortage.values())
        + pulp.lpSum(idle_cost * excess_var for excess_var in excess.values())
    )

    for zone in zones:
        inbound = pulp.lpSum(x[(i, zone)] for i in zones if i != zone)
        outbound = pulp.lpSum(x[(zone, j)] for j in zones if j != zone)
        model += current_inventory[zone] + inbound - outbound + shortage[zone] - excess[zone] == target_lookup[zone]
        model += outbound <= current_inventory[zone]

    model += pulp.lpSum(x.values()) <= move_capacity
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    move_rows = []
    for (origin, destination), variable in x.items():
        move_rows.append({"origin": origin, "destination": destination, "bikes_moved": int(round(variable.value() or 0))})

    summary_rows = []
    for zone in zones:
        inbound = sum(int(round(x[(origin, zone)].value() or 0)) for origin in zones if origin != zone)
        outbound = sum(int(round(x[(zone, dest)].value() or 0)) for dest in zones if dest != zone)
        post_move_inventory = current_inventory[zone] + inbound - outbound
        summary_rows.append(
            {
                "zone": zone,
                "current_inventory": current_inventory[zone],
                "target_inventory": target_lookup[zone],
                "inbound_bikes": inbound,
                "outbound_bikes": outbound,
                "post_move_inventory": post_move_inventory,
                "shortage_slack": int(round(shortage[zone].value() or 0)),
                "excess_slack": int(round(excess[zone].value() or 0)),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(move_rows)


def run_zone_sensitivity(
    zone_targets: pd.DataFrame,
    current_inventory: Dict[str, int],
    move_cost_values: Iterable[float],
    shortage_cost: float,
    idle_cost: float,
    move_capacity: int,
) -> pd.DataFrame:
    rows = []
    for move_cost in move_cost_values:
        summary, moves = solve_zone_rebalancing(
            zone_targets=zone_targets,
            current_inventory=current_inventory,
            move_cost=move_cost,
            shortage_cost=shortage_cost,
            idle_cost=idle_cost,
            move_capacity=move_capacity,
        )
        rows.append(
            {
                "move_cost": move_cost,
                "total_bikes_moved": moves["bikes_moved"].sum(),
                "total_shortage_slack": summary["shortage_slack"].sum(),
                "total_excess_slack": summary["excess_slack"].sum(),
            }
        )
    return pd.DataFrame(rows)

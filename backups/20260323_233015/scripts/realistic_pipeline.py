#!/usr/bin/env python3
"""Rebuild and solve the realistic childcare expansion model.

This script intentionally leaves the Part 1 notebooks mostly untouched and
provides a cleaner Part 2 pipeline with two concrete repairs:

1. Zip codes with zero child population get zero total demand instead of a
   spurious requirement of one slot.
2. The 20% realistic expansion cap is implemented from cumulative floors, so
   the three blocks sum to floor(0.20 * n_f) instead of losing capacity to
   separate floor operations.

The default solve mode keeps the additive block-cost formulation used in the
current notebook implementation because it solves cleanly and reproducibly.
An optional assignment-style piecewise-total mode is also available for
comparison, but it is slower.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only outside the Gurobi env
    gp = None
    GRB = None


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor that looks like the project root."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing data/ and results/.")


def standardize_zipcode(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )


def haversine_miles(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in miles."""
    radius_miles = 3958.7613
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return radius_miles * c


def build_realistic_inputs(project_root: Path) -> dict[str, int]:
    """Create realistic-model input tables from the shared/ideal outputs."""
    shared_dir = project_root / "data" / "processed" / "shared"
    ideal_dir = project_root / "data" / "processed" / "ideal"
    realistic_dir = project_root / "data" / "processed" / "realistic"
    realistic_dir.mkdir(parents=True, exist_ok=True)

    fac_clean = pd.read_csv(shared_dir / "clean_childcare_facilities.csv")
    facility_geo = pd.read_csv(shared_dir / "facility_geo_ready.csv")
    candidate_geo = pd.read_csv(shared_dir / "potential_locations_geo_ready.csv")
    zip_df = pd.read_csv(ideal_dir / "zipcode_demand_supply_ideal.csv")
    build_options = pd.read_csv(ideal_dir / "build_options_ideal.csv")

    for df in [fac_clean, facility_geo, candidate_geo, zip_df]:
        if "zipcode" in df.columns:
            df["zipcode"] = standardize_zipcode(df["zipcode"])

    facility_geo["facility_id"] = facility_geo["facility_id"].astype(str)
    fac_clean["facility_id"] = fac_clean["facility_id"].astype(str)
    candidate_geo["candidate_id"] = (
        pd.to_numeric(candidate_geo["candidate_id"], errors="coerce").astype("Int64")
    )

    fac_existing = fac_clean.copy()
    fac_existing = fac_existing[fac_existing["is_active_facility"] == 1].copy()
    fac_existing = fac_existing[fac_existing["total_capacity"].fillna(0) > 0].copy()
    fac_existing["n_f"] = pd.to_numeric(fac_existing["total_capacity"], errors="coerce").fillna(0).astype(int)
    fac_existing["b_f"] = pd.to_numeric(
        fac_existing["under5_capacity_current"], errors="coerce"
    ).fillna(0).astype(int)

    geo_cols = ["facility_id", "latitude", "longitude", "zipcode"]
    fac_existing = fac_existing.merge(
        facility_geo[geo_cols],
        on="facility_id",
        how="left",
        suffixes=("", "_geo"),
    )
    fac_existing["zipcode"] = fac_existing["zipcode"].fillna(fac_existing["zipcode_geo"])
    fac_existing = fac_existing.drop(columns=["zipcode_geo"], errors="ignore")

    fac_existing["cap10_cum"] = np.floor(0.10 * fac_existing["n_f"]).astype(int)
    fac_existing["cap15_cum"] = np.floor(0.15 * fac_existing["n_f"]).astype(int)
    fac_existing["cap20_cum"] = np.floor(0.20 * fac_existing["n_f"]).astype(int)

    fac_existing["cap1"] = fac_existing["cap10_cum"]
    fac_existing["cap2"] = (
        fac_existing["cap15_cum"] - fac_existing["cap10_cum"]
    ).clip(lower=0).astype(int)
    fac_existing["cap3"] = (
        fac_existing["cap20_cum"] - fac_existing["cap15_cum"]
    ).clip(lower=0).astype(int)
    fac_existing["U_f_realistic"] = fac_existing["cap20_cum"]

    fac_existing["coef1"] = (20000 + 200 * fac_existing["n_f"]) / fac_existing["n_f"]
    fac_existing["coef2"] = (20000 + 400 * fac_existing["n_f"]) / fac_existing["n_f"]
    fac_existing["coef3"] = (20000 + 1000 * fac_existing["n_f"]) / fac_existing["n_f"]

    facility_expansion_params_realistic = fac_existing[
        [
            "facility_id",
            "facility_name",
            "program_type",
            "zipcode",
            "latitude",
            "longitude",
            "n_f",
            "b_f",
            "cap1",
            "cap2",
            "cap3",
            "cap10_cum",
            "cap15_cum",
            "cap20_cum",
            "U_f_realistic",
            "coef1",
            "coef2",
            "coef3",
        ]
    ].copy()

    zipcode_demand_supply_realistic = zip_df.copy()
    zero_child_mask = zipcode_demand_supply_realistic["child_pop_0_12"].fillna(0) <= 0
    zero_child_req_total_fixed = int(
        (
            zero_child_mask
            & (zipcode_demand_supply_realistic["req_total"].fillna(0) > 0)
        ).sum()
    )

    zipcode_demand_supply_realistic.loc[zero_child_mask, "req_total"] = 0
    zipcode_demand_supply_realistic["gap_total"] = (
        zipcode_demand_supply_realistic["req_total"]
        - zipcode_demand_supply_realistic["existing_total"]
    ).clip(lower=0)
    zipcode_demand_supply_realistic["is_desert_total_before"] = (
        zipcode_demand_supply_realistic["gap_total"] > 0
    ).astype(int)
    zipcode_demand_supply_realistic["needs_intervention_before"] = (
        (zipcode_demand_supply_realistic["gap_total"] > 0)
        | (zipcode_demand_supply_realistic["gap_under5"] > 0)
    ).astype(int)

    candidate_geo = candidate_geo.copy()
    candidate_geo["zipcode"] = standardize_zipcode(candidate_geo["zipcode"])

    candidate_build_options_realistic = (
        candidate_geo.assign(key=1)
        .merge(build_options.assign(key=1), on="key", how="inner")
        .drop(columns=["key"])
    )

    min_dist = 0.06

    cand_exist_pairs = candidate_geo.merge(
        facility_expansion_params_realistic[["facility_id", "zipcode", "latitude", "longitude"]],
        on="zipcode",
        how="inner",
        suffixes=("_cand", "_exist"),
    )
    cand_exist_pairs["distance_miles"] = haversine_miles(
        cand_exist_pairs["latitude_cand"],
        cand_exist_pairs["longitude_cand"],
        cand_exist_pairs["latitude_exist"],
        cand_exist_pairs["longitude_exist"],
    )
    candidate_existing_conflicts = cand_exist_pairs[
        cand_exist_pairs["distance_miles"] < min_dist
    ][["candidate_id", "facility_id", "zipcode", "distance_miles"]].drop_duplicates()

    cand1 = candidate_geo.rename(
        columns={
            "candidate_id": "candidate_id_1",
            "latitude": "latitude_1",
            "longitude": "longitude_1",
        }
    )
    cand2 = candidate_geo.rename(
        columns={
            "candidate_id": "candidate_id_2",
            "latitude": "latitude_2",
            "longitude": "longitude_2",
        }
    )
    candidate_candidate_pairs = cand1.merge(cand2, on="zipcode", how="inner")
    candidate_candidate_pairs = candidate_candidate_pairs[
        candidate_candidate_pairs["candidate_id_1"] < candidate_candidate_pairs["candidate_id_2"]
    ].copy()
    candidate_candidate_pairs["distance_miles"] = haversine_miles(
        candidate_candidate_pairs["latitude_1"],
        candidate_candidate_pairs["longitude_1"],
        candidate_candidate_pairs["latitude_2"],
        candidate_candidate_pairs["longitude_2"],
    )
    candidate_candidate_conflicts = candidate_candidate_pairs[
        candidate_candidate_pairs["distance_miles"] < min_dist
    ][["candidate_id_1", "candidate_id_2", "zipcode", "distance_miles"]].drop_duplicates()

    exist1 = facility_expansion_params_realistic.rename(
        columns={
            "facility_id": "facility_id_1",
            "latitude": "latitude_1",
            "longitude": "longitude_1",
        }
    )
    exist2 = facility_expansion_params_realistic.rename(
        columns={
            "facility_id": "facility_id_2",
            "latitude": "latitude_2",
            "longitude": "longitude_2",
        }
    )
    existing_existing_pairs = exist1.merge(exist2, on="zipcode", how="inner")
    existing_existing_pairs = existing_existing_pairs[
        existing_existing_pairs["facility_id_1"] < existing_existing_pairs["facility_id_2"]
    ].copy()
    existing_existing_pairs["distance_miles"] = haversine_miles(
        existing_existing_pairs["latitude_1"],
        existing_existing_pairs["longitude_1"],
        existing_existing_pairs["latitude_2"],
        existing_existing_pairs["longitude_2"],
    )
    existing_existing_too_close = existing_existing_pairs[
        existing_existing_pairs["distance_miles"] < min_dist
    ][["facility_id_1", "facility_id_2", "zipcode", "distance_miles"]].drop_duplicates()

    blocked_candidates = (
        candidate_existing_conflicts[["candidate_id"]]
        .drop_duplicates()
        .assign(blocked_by_existing=1)
    )
    candidate_geo_realistic = candidate_geo.merge(
        blocked_candidates, on="candidate_id", how="left"
    )
    candidate_geo_realistic["blocked_by_existing"] = (
        candidate_geo_realistic["blocked_by_existing"].fillna(0).astype(int)
    )

    candidate_build_options_realistic = candidate_build_options_realistic.merge(
        candidate_geo_realistic[["candidate_id", "blocked_by_existing"]],
        on="candidate_id",
        how="left",
    )
    candidate_build_options_realistic["blocked_by_existing"] = (
        candidate_build_options_realistic["blocked_by_existing"].fillna(0).astype(int)
    )
    candidate_build_options_feasible = candidate_build_options_realistic[
        candidate_build_options_realistic["blocked_by_existing"] == 0
    ].copy()

    realistic_parameter_summary = pd.DataFrame(
        {
            "metric": [
                "n_zipcodes",
                "n_existing_facilities_expandable",
                "citywide_existing_total_capacity",
                "citywide_existing_under5_capacity",
                "citywide_realistic_expansion_capacity",
                "n_candidate_sites",
                "n_blocked_candidate_sites",
                "n_feasible_candidate_sites",
                "n_candidate_candidate_conflicts",
                "n_candidate_existing_conflicts",
                "n_existing_existing_too_close_diagnostic",
                "n_zero_child_zipcodes",
                "n_zero_child_req_total_fixed",
                "n_existing_facilities_missing_coords",
            ],
            "value": [
                zipcode_demand_supply_realistic["zipcode"].nunique(),
                facility_expansion_params_realistic["facility_id"].nunique(),
                facility_expansion_params_realistic["n_f"].sum(),
                facility_expansion_params_realistic["b_f"].sum(),
                facility_expansion_params_realistic["U_f_realistic"].sum(),
                candidate_geo_realistic["candidate_id"].nunique(),
                candidate_geo_realistic["blocked_by_existing"].sum(),
                (candidate_geo_realistic["blocked_by_existing"] == 0).sum(),
                len(candidate_candidate_conflicts),
                len(candidate_existing_conflicts),
                len(existing_existing_too_close),
                int(zero_child_mask.sum()),
                zero_child_req_total_fixed,
                int(facility_expansion_params_realistic["latitude"].isna().sum()),
            ],
        }
    )

    realistic_assumptions = {
        "distance_threshold_miles": 0.06,
        "distance_enforced_within_zipcode_only": True,
        "existing_existing_conflicts_are_diagnostic_only": True,
        "candidate_existing_conflict_blocks_candidate": True,
        "use_under5_capacity_current_for_existing_capacity": True,
        "new_build_costs_and_capacities_reused_from_ideal_model": True,
        "realistic_expansion_cap_implemented_as_floor_20pct_total": True,
        "expansion_block_caps_from_cumulative_floors": True,
        "zero_child_zipcodes_have_zero_total_requirement": True,
        "default_cost_mode": "additive_blocks",
        "optional_cost_mode": "assignment_piecewise_total",
    }

    facility_expansion_params_realistic.to_csv(
        realistic_dir / "facility_expansion_params_realistic.csv", index=False
    )
    zipcode_demand_supply_realistic.to_csv(
        realistic_dir / "zipcode_demand_supply_realistic.csv", index=False
    )
    candidate_geo_realistic.to_csv(
        realistic_dir / "candidate_sites_realistic.csv", index=False
    )
    candidate_build_options_feasible.to_csv(
        realistic_dir / "candidate_build_options_realistic.csv", index=False
    )
    candidate_existing_conflicts.to_csv(
        realistic_dir / "candidate_existing_conflicts.csv", index=False
    )
    candidate_candidate_conflicts.to_csv(
        realistic_dir / "candidate_candidate_conflicts.csv", index=False
    )
    existing_existing_too_close.to_csv(
        realistic_dir / "existing_existing_too_close_diagnostic.csv", index=False
    )
    realistic_parameter_summary.to_csv(
        realistic_dir / "realistic_parameter_summary.csv", index=False
    )
    with open(realistic_dir / "assumptions_realistic.json", "w", encoding="utf-8") as f:
        json.dump(realistic_assumptions, f, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {"assumption": list(realistic_assumptions.keys()), "value": list(realistic_assumptions.values())}
    ).to_csv(realistic_dir / "assumptions_realistic.csv", index=False)

    return {
        "zero_child_req_total_fixed": zero_child_req_total_fixed,
        "citywide_realistic_expansion_capacity": int(
            facility_expansion_params_realistic["U_f_realistic"].sum()
        ),
        "n_feasible_candidate_sites": int(
            (candidate_geo_realistic["blocked_by_existing"] == 0).sum()
        ),
    }


def _build_additive_model(model: gp.Model, fac_df: pd.DataFrame):
    facility_ids = fac_df["facility_id"].tolist()
    cap1 = dict(zip(fac_df["facility_id"], fac_df["cap1"]))
    cap2 = dict(zip(fac_df["facility_id"], fac_df["cap2"]))
    cap3 = dict(zip(fac_df["facility_id"], fac_df["cap3"]))
    coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
    coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
    coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

    x_low = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="x_low")
    x_mid = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="x_mid")
    x_high = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="x_high")
    u = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="u")

    for facility_id in facility_ids:
        model.addConstr(x_low[facility_id] <= cap1[facility_id], name=f"cap1_{facility_id}")
        model.addConstr(x_mid[facility_id] <= cap2[facility_id], name=f"cap2_{facility_id}")
        model.addConstr(x_high[facility_id] <= cap3[facility_id], name=f"cap3_{facility_id}")
        model.addConstr(
            u[facility_id] <= x_low[facility_id] + x_mid[facility_id] + x_high[facility_id],
            name=f"under5_expand_cap_{facility_id}",
        )

    return {
        "mode": "additive_blocks",
        "x_low": x_low,
        "x_mid": x_mid,
        "x_high": x_high,
        "u": u,
        "expand_total_expr": {
            facility_id: x_low[facility_id] + x_mid[facility_id] + x_high[facility_id]
            for facility_id in facility_ids
        },
        "expansion_cost_expr": gp.quicksum(
            coef1[facility_id] * x_low[facility_id]
            + coef2[facility_id] * x_mid[facility_id]
            + coef3[facility_id] * x_high[facility_id]
            for facility_id in facility_ids
        ),
    }


def _build_assignment_piecewise_model(model: gp.Model, fac_df: pd.DataFrame):
    facility_ids = fac_df["facility_id"].tolist()
    cap10 = dict(zip(fac_df["facility_id"], fac_df["cap10_cum"]))
    cap15 = dict(zip(fac_df["facility_id"], fac_df["cap15_cum"]))
    cap20 = dict(zip(fac_df["facility_id"], fac_df["cap20_cum"]))
    coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
    coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
    coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

    x_total = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="x_total")
    u = model.addVars(facility_ids, vtype=GRB.INTEGER, lb=0, name="u")
    z_low = model.addVars(facility_ids, vtype=GRB.BINARY, name="z_low")
    z_mid = model.addVars(facility_ids, vtype=GRB.BINARY, name="z_mid")
    z_high = model.addVars(facility_ids, vtype=GRB.BINARY, name="z_high")
    expansion_cost = model.addVars(facility_ids, lb=0.0, name="expansion_cost")

    for facility_id in facility_ids:
        model.addConstr(
            z_low[facility_id] + z_mid[facility_id] + z_high[facility_id] <= 1,
            name=f"one_band_{facility_id}",
        )
        model.addConstr(
            x_total[facility_id]
            <= cap10[facility_id] * z_low[facility_id]
            + cap15[facility_id] * z_mid[facility_id]
            + cap20[facility_id] * z_high[facility_id],
            name=f"band_upper_{facility_id}",
        )
        model.addConstr(
            x_total[facility_id]
            >= z_low[facility_id]
            + (cap10[facility_id] + 1) * z_mid[facility_id]
            + (cap15[facility_id] + 1) * z_high[facility_id],
            name=f"band_lower_{facility_id}",
        )
        model.addConstr(u[facility_id] <= x_total[facility_id], name=f"under5_cap_{facility_id}")

        if cap10[facility_id] == 0:
            model.addConstr(z_low[facility_id] == 0, name=f"disable_low_{facility_id}")
        if cap15[facility_id] <= cap10[facility_id]:
            model.addConstr(z_mid[facility_id] == 0, name=f"disable_mid_{facility_id}")
        if cap20[facility_id] <= cap15[facility_id]:
            model.addConstr(z_high[facility_id] == 0, name=f"disable_high_{facility_id}")

        max_cost = coef3[facility_id] * cap20[facility_id]
        model.addConstr(
            expansion_cost[facility_id]
            >= coef1[facility_id] * x_total[facility_id] - max_cost * (1 - z_low[facility_id]),
            name=f"low_cost_{facility_id}",
        )
        model.addConstr(
            expansion_cost[facility_id]
            >= coef2[facility_id] * x_total[facility_id] - max_cost * (1 - z_mid[facility_id]),
            name=f"mid_cost_{facility_id}",
        )
        model.addConstr(
            expansion_cost[facility_id]
            >= coef3[facility_id] * x_total[facility_id] - max_cost * (1 - z_high[facility_id]),
            name=f"high_cost_{facility_id}",
        )

    return {
        "mode": "assignment_piecewise_total",
        "x_total": x_total,
        "u": u,
        "z_low": z_low,
        "z_mid": z_mid,
        "z_high": z_high,
        "expansion_cost_var": expansion_cost,
        "expansion_cost_expr": gp.quicksum(expansion_cost[facility_id] for facility_id in facility_ids),
    }


def solve_realistic_model(
    project_root: Path,
    cost_mode: str,
    time_limit: int,
    mip_gap: float,
    output_flag: int,
) -> dict[str, float | int | str]:
    """Solve the realistic MILP and export result tables."""
    if gp is None:
        raise ImportError("gurobipy is not available. Run this script from the Gurobi conda environment.")

    realistic_dir = project_root / "data" / "processed" / "realistic"
    results_dir = project_root / "results" / "realistic"
    solutions_dir = results_dir / "solutions"
    logs_dir = results_dir / "logs"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    zip_df = pd.read_csv(realistic_dir / "zipcode_demand_supply_realistic.csv")
    fac_df = pd.read_csv(realistic_dir / "facility_expansion_params_realistic.csv")
    cand_df = pd.read_csv(realistic_dir / "candidate_build_options_realistic.csv")
    cc_conflict_df = pd.read_csv(realistic_dir / "candidate_candidate_conflicts.csv")

    for df in [zip_df, fac_df, cand_df]:
        df["zipcode"] = standardize_zipcode(df["zipcode"])
    fac_df["facility_id"] = fac_df["facility_id"].astype(str)
    cand_df["candidate_id"] = pd.to_numeric(cand_df["candidate_id"], errors="coerce").astype(int)
    cand_df["size"] = cand_df["size"].astype(str)

    if not cc_conflict_df.empty:
        cc_conflict_df["zipcode"] = standardize_zipcode(cc_conflict_df["zipcode"])
        cc_conflict_df["candidate_id_1"] = pd.to_numeric(
            cc_conflict_df["candidate_id_1"], errors="coerce"
        ).astype(int)
        cc_conflict_df["candidate_id_2"] = pd.to_numeric(
            cc_conflict_df["candidate_id_2"], errors="coerce"
        ).astype(int)

    facility_ids = fac_df["facility_id"].tolist()
    zipcodes = zip_df["zipcode"].tolist()
    candidate_sites = sorted(cand_df["candidate_id"].unique().tolist())
    candidate_size_keys = list(zip(cand_df["candidate_id"], cand_df["size"]))

    req_total = dict(zip(zip_df["zipcode"], zip_df["req_total"]))
    req_under5 = dict(zip(zip_df["zipcode"], zip_df["req_under5"]))
    existing_total = dict(zip(zip_df["zipcode"], zip_df["existing_total"]))
    existing_under5 = dict(zip(zip_df["zipcode"], zip_df["existing_under5"]))

    build_total = {
        (row["candidate_id"], row["size"]): row["new_total_capacity"]
        for _, row in cand_df.iterrows()
    }
    build_under5_max = {
        (row["candidate_id"], row["size"]): row["new_under5_capacity_max"]
        for _, row in cand_df.iterrows()
    }
    build_cost = {
        (row["candidate_id"], row["size"]): row["fixed_build_cost"]
        for _, row in cand_df.iterrows()
    }

    fac_by_zip = fac_df.groupby("zipcode")["facility_id"].apply(list).to_dict()
    candsize_by_zip = (
        cand_df.groupby("zipcode")[["candidate_id", "size"]]
        .apply(lambda g: list(map(tuple, g.to_numpy())))
        .to_dict()
    )
    sizes_by_candidate = cand_df.groupby("candidate_id")["size"].apply(list).to_dict()

    log_file = logs_dir / "realistic_model.log"
    model = gp.Model("realistic_childcare")
    model.Params.LogFile = str(log_file)
    model.Params.MIPGap = mip_gap
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = output_flag
    model.Params.Presolve = 2
    model.Params.Threads = 0

    y = model.addVars(candidate_size_keys, vtype=GRB.BINARY, name="y")
    g = model.addVars(candidate_size_keys, vtype=GRB.INTEGER, lb=0, name="g")

    for candidate_id in candidate_sites:
        candidate_sizes = sizes_by_candidate.get(candidate_id, [])
        if candidate_sizes:
            model.addConstr(
                gp.quicksum(y[candidate_id, size] for size in candidate_sizes) <= 1,
                name=f"one_size_per_candidate_{candidate_id}",
            )

    for candidate_id, size in candidate_size_keys:
        model.addConstr(
            g[candidate_id, size] <= build_under5_max[(candidate_id, size)] * y[candidate_id, size],
            name=f"new_under5_cap_{candidate_id}_{size}",
        )

    if not cc_conflict_df.empty:
        for _, row in cc_conflict_df.iterrows():
            candidate_1 = int(row["candidate_id_1"])
            candidate_2 = int(row["candidate_id_2"])
            sizes_1 = sizes_by_candidate.get(candidate_1, [])
            sizes_2 = sizes_by_candidate.get(candidate_2, [])
            if sizes_1 and sizes_2:
                model.addConstr(
                    gp.quicksum(y[candidate_1, size] for size in sizes_1)
                    + gp.quicksum(y[candidate_2, size] for size in sizes_2)
                    <= 1,
                    name=f"cc_conflict_{candidate_1}_{candidate_2}",
                )

    if cost_mode == "additive_blocks":
        model_artifacts = _build_additive_model(model, fac_df)
        expand_total_expr = model_artifacts["expand_total_expr"]
        u = model_artifacts["u"]
        expansion_cost_expr = model_artifacts["expansion_cost_expr"]
    elif cost_mode == "assignment_piecewise_total":
        model_artifacts = _build_assignment_piecewise_model(model, fac_df)
        expand_total_expr = model_artifacts["x_total"]
        u = model_artifacts["u"]
        expansion_cost_expr = model_artifacts["expansion_cost_expr"]
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")

    for zipcode in zipcodes:
        model.addConstr(
            existing_total[zipcode]
            + gp.quicksum(expand_total_expr[facility_id] for facility_id in fac_by_zip.get(zipcode, []))
            + gp.quicksum(build_total[(candidate_id, size)] * y[candidate_id, size] for candidate_id, size in candsize_by_zip.get(zipcode, []))
            >= req_total[zipcode],
            name=f"total_req_{zipcode}",
        )
        model.addConstr(
            existing_under5[zipcode]
            + gp.quicksum(u[facility_id] for facility_id in fac_by_zip.get(zipcode, []))
            + gp.quicksum(g[candidate_id, size] for candidate_id, size in candsize_by_zip.get(zipcode, []))
            >= req_under5[zipcode],
            name=f"under5_req_{zipcode}",
        )

    special_equipment_cost = 100
    new_build_cost_expr = gp.quicksum(
        build_cost[(candidate_id, size)] * y[candidate_id, size]
        for candidate_id, size in candidate_size_keys
    )
    under5_equipment_cost_expr = (
        gp.quicksum(special_equipment_cost * u[facility_id] for facility_id in facility_ids)
        + gp.quicksum(special_equipment_cost * g[candidate_id, size] for candidate_id, size in candidate_size_keys)
    )

    model.setObjective(
        expansion_cost_expr + new_build_cost_expr + under5_equipment_cost_expr,
        GRB.MINIMIZE,
    )

    model.update()
    start_time = time.time()
    model.optimize()
    solve_seconds = time.time() - start_time

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0

    if not has_solution:
        raise RuntimeError(f"Realistic model finished without a usable solution (status {model.Status}).")

    facility_solution = fac_df.copy()
    if cost_mode == "additive_blocks":
        x_low = model_artifacts["x_low"]
        x_mid = model_artifacts["x_mid"]
        x_high = model_artifacts["x_high"]
        facility_solution["expand_low_band"] = facility_solution["facility_id"].map(lambda f: x_low[f].X)
        facility_solution["expand_mid_band"] = facility_solution["facility_id"].map(lambda f: x_mid[f].X)
        facility_solution["expand_high_band"] = facility_solution["facility_id"].map(lambda f: x_high[f].X)
        facility_solution["selected_cost_band"] = np.select(
            [
                facility_solution["expand_high_band"] > 1e-6,
                facility_solution["expand_mid_band"] > 1e-6,
                facility_solution["expand_low_band"] > 1e-6,
            ],
            ["15_20_pct", "10_15_pct", "0_10_pct"],
            default="no_expansion",
        )
    else:
        x_total = model_artifacts["x_total"]
        z_low = model_artifacts["z_low"]
        z_mid = model_artifacts["z_mid"]
        z_high = model_artifacts["z_high"]
        facility_solution["expand_low_band"] = facility_solution["facility_id"].map(
            lambda f: x_total[f].X if z_low[f].X > 0.5 else 0.0
        )
        facility_solution["expand_mid_band"] = facility_solution["facility_id"].map(
            lambda f: x_total[f].X if z_mid[f].X > 0.5 else 0.0
        )
        facility_solution["expand_high_band"] = facility_solution["facility_id"].map(
            lambda f: x_total[f].X if z_high[f].X > 0.5 else 0.0
        )
        facility_solution["selected_cost_band"] = np.select(
            [
                facility_solution["expand_high_band"] > 1e-6,
                facility_solution["expand_mid_band"] > 1e-6,
                facility_solution["expand_low_band"] > 1e-6,
            ],
            ["15_20_pct", "10_15_pct", "0_10_pct"],
            default="no_expansion",
        )

    facility_solution["u_under5"] = facility_solution["facility_id"].map(lambda f: u[f].X)
    facility_solution["expand_total"] = (
        facility_solution["expand_low_band"]
        + facility_solution["expand_mid_band"]
        + facility_solution["expand_high_band"]
    )
    if cost_mode == "additive_blocks":
        facility_solution["expansion_cost"] = (
            facility_solution["coef1"] * facility_solution["expand_low_band"]
            + facility_solution["coef2"] * facility_solution["expand_mid_band"]
            + facility_solution["coef3"] * facility_solution["expand_high_band"]
        )
    else:
        expansion_cost_var = model_artifacts["expansion_cost_var"]
        facility_solution["expansion_cost"] = facility_solution["facility_id"].map(lambda f: expansion_cost_var[f].X)
    facility_solution["under5_equipment_cost_expansion"] = 100 * facility_solution["u_under5"]
    facility_solution["total_expansion_related_cost"] = (
        facility_solution["expansion_cost"] + facility_solution["under5_equipment_cost_expansion"]
    )
    facility_solution = facility_solution[facility_solution["expand_total"] > 1e-6].copy()

    build_rows: list[dict[str, object]] = []
    for candidate_id, size in candidate_size_keys:
        if y[candidate_id, size].X > 0.5:
            row = cand_df[
                (cand_df["candidate_id"] == candidate_id) & (cand_df["size"] == size)
            ].iloc[0].to_dict()
            row["build_selected"] = int(round(y[candidate_id, size].X))
            row["new_under5_assigned"] = g[candidate_id, size].X
            row["under5_equipment_cost_newbuild"] = 100 * g[candidate_id, size].X
            row["total_newbuild_related_cost"] = (
                row["fixed_build_cost"] + row["under5_equipment_cost_newbuild"]
            )
            build_rows.append(row)
    new_build_solution = pd.DataFrame(build_rows)

    if not facility_solution.empty:
        facility_expand_zip = (
            facility_solution.groupby("zipcode", as_index=False)
            .agg(
                expanded_total=("expand_total", "sum"),
                expanded_under5=("u_under5", "sum"),
                expansion_cost=("expansion_cost", "sum"),
                under5_equipment_cost_expansion=("under5_equipment_cost_expansion", "sum"),
                total_expansion_related_cost=("total_expansion_related_cost", "sum"),
            )
        )
    else:
        facility_expand_zip = pd.DataFrame(
            columns=[
                "zipcode",
                "expanded_total",
                "expanded_under5",
                "expansion_cost",
                "under5_equipment_cost_expansion",
                "total_expansion_related_cost",
            ]
        )

    if not new_build_solution.empty:
        new_build_zip = (
            new_build_solution.groupby("zipcode", as_index=False)
            .agg(
                n_new_facilities=("candidate_id", "count"),
                new_total_capacity=("new_total_capacity", "sum"),
                new_under5_capacity=("new_under5_assigned", "sum"),
                new_build_cost=("fixed_build_cost", "sum"),
                under5_equipment_cost_newbuild=("under5_equipment_cost_newbuild", "sum"),
                total_newbuild_related_cost=("total_newbuild_related_cost", "sum"),
            )
        )
        build_size_summary = (
            new_build_solution.groupby("size", as_index=False)
            .agg(
                n_new_facilities=("candidate_id", "count"),
                total_capacity=("new_total_capacity", "sum"),
                total_under5=("new_under5_assigned", "sum"),
                fixed_build_cost=("fixed_build_cost", "sum"),
            )
            .sort_values("size")
        )
    else:
        new_build_zip = pd.DataFrame(
            columns=[
                "zipcode",
                "n_new_facilities",
                "new_total_capacity",
                "new_under5_capacity",
                "new_build_cost",
                "under5_equipment_cost_newbuild",
                "total_newbuild_related_cost",
            ]
        )
        build_size_summary = pd.DataFrame(
            columns=["size", "n_new_facilities", "total_capacity", "total_under5", "fixed_build_cost"]
        )

    zipcode_solution = zip_df.copy()
    zipcode_solution = zipcode_solution.merge(facility_expand_zip, on="zipcode", how="left")
    zipcode_solution = zipcode_solution.merge(new_build_zip, on="zipcode", how="left")
    for col in [
        "expanded_total",
        "expanded_under5",
        "expansion_cost",
        "under5_equipment_cost_expansion",
        "total_expansion_related_cost",
        "n_new_facilities",
        "new_total_capacity",
        "new_under5_capacity",
        "new_build_cost",
        "under5_equipment_cost_newbuild",
        "total_newbuild_related_cost",
    ]:
        zipcode_solution[col] = zipcode_solution[col].fillna(0)

    zipcode_solution["final_total_capacity"] = (
        zipcode_solution["existing_total"]
        + zipcode_solution["expanded_total"]
        + zipcode_solution["new_total_capacity"]
    )
    zipcode_solution["final_under5_capacity"] = (
        zipcode_solution["existing_under5"]
        + zipcode_solution["expanded_under5"]
        + zipcode_solution["new_under5_capacity"]
    )
    zipcode_solution["total_slack"] = zipcode_solution["final_total_capacity"] - zipcode_solution["req_total"]
    zipcode_solution["under5_slack"] = zipcode_solution["final_under5_capacity"] - zipcode_solution["req_under5"]
    zipcode_solution["all_requirements_met_after"] = (
        (zipcode_solution["total_slack"] >= -1e-6)
        & (zipcode_solution["under5_slack"] >= -1e-6)
    ).astype(int)

    objective_breakdown = pd.DataFrame(
        {
            "component": [
                "expansion_cost",
                "new_build_cost",
                "under5_equipment_cost",
                "total_objective",
            ],
            "value": [
                expansion_cost_expr.getValue(),
                new_build_cost_expr.getValue(),
                under5_equipment_cost_expr.getValue(),
                model.ObjVal,
            ],
        }
    )

    size_counts = (
        new_build_solution["size"].value_counts().to_dict()
        if not new_build_solution.empty
        else {}
    )
    run_metadata = pd.DataFrame(
        {
            "metric": [
                "model_status_code",
                "model_status_text",
                "objective_value",
                "best_bound",
                "mip_gap",
                "solve_seconds",
                "num_variables",
                "num_constraints",
                "solution_count",
                "cost_mode",
                "time_limit_seconds",
            ],
            "value": [
                model.Status,
                status_map.get(model.Status, str(model.Status)),
                model.ObjVal,
                model.ObjBound,
                model.MIPGap if model.IsMIP else 0.0,
                solve_seconds,
                model.NumVars,
                model.NumConstrs,
                model.SolCount,
                cost_mode,
                time_limit,
            ],
        }
    )
    summary_stats = pd.DataFrame(
        {
            "metric": [
                "n_expanded_facilities",
                "total_expansion_slots",
                "total_expansion_under5_slots",
                "n_new_facilities",
                "total_new_capacity",
                "total_new_under5_capacity",
                "n_new_small_facilities",
                "n_new_medium_facilities",
                "n_new_large_facilities",
                "zipcodes_meeting_all_requirements_after",
            ],
            "value": [
                len(facility_solution),
                facility_solution["expand_total"].sum() if not facility_solution.empty else 0,
                facility_solution["u_under5"].sum() if not facility_solution.empty else 0,
                len(new_build_solution) if not new_build_solution.empty else 0,
                new_build_solution["new_total_capacity"].sum() if not new_build_solution.empty else 0,
                new_build_solution["new_under5_assigned"].sum() if not new_build_solution.empty else 0,
                size_counts.get("small", 0),
                size_counts.get("medium", 0),
                size_counts.get("large", 0),
                zipcode_solution["all_requirements_met_after"].sum(),
            ],
        }
    )

    facility_solution.to_csv(solutions_dir / "facility_solution_realistic.csv", index=False)
    zipcode_solution.to_csv(solutions_dir / "zipcode_solution_realistic.csv", index=False)
    objective_breakdown.to_csv(solutions_dir / "objective_breakdown_realistic.csv", index=False)
    run_metadata.to_csv(solutions_dir / "run_metadata_realistic.csv", index=False)
    summary_stats.to_csv(solutions_dir / "summary_stats_realistic.csv", index=False)
    build_size_summary.to_csv(solutions_dir / "build_size_summary_realistic.csv", index=False)
    if not new_build_solution.empty:
        new_build_solution.to_csv(solutions_dir / "new_build_solution_realistic.csv", index=False)
    else:
        pd.DataFrame(
            columns=cand_df.columns.tolist()
            + [
                "build_selected",
                "new_under5_assigned",
                "under5_equipment_cost_newbuild",
                "total_newbuild_related_cost",
            ]
        ).to_csv(solutions_dir / "new_build_solution_realistic.csv", index=False)

    return {
        "status": status_map.get(model.Status, str(model.Status)),
        "objective_value": float(model.ObjVal),
        "solve_seconds": float(solve_seconds),
        "cost_mode": cost_mode,
        "n_new_facilities": int(len(new_build_solution)),
        "n_expanded_facilities": int(len(facility_solution)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="rebuild realistic input tables")
    prepare_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="path to the project root (defaults to auto-detection)",
    )

    solve_parser = subparsers.add_parser("solve", help="solve the realistic optimization model")
    solve_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="path to the project root (defaults to auto-detection)",
    )
    solve_parser.add_argument(
        "--cost-mode",
        choices=["additive_blocks", "assignment_piecewise_total"],
        default="additive_blocks",
        help="realistic expansion cost formulation",
    )
    solve_parser.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit in seconds")
    solve_parser.add_argument("--mip-gap", type=float, default=0.001, help="relative MIP gap tolerance")
    solve_parser.add_argument("--quiet", action="store_true", help="suppress Gurobi console output")

    all_parser = subparsers.add_parser("all", help="prepare inputs and solve the realistic model")
    all_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="path to the project root (defaults to auto-detection)",
    )
    all_parser.add_argument(
        "--cost-mode",
        choices=["additive_blocks", "assignment_piecewise_total"],
        default="additive_blocks",
        help="realistic expansion cost formulation",
    )
    all_parser.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit in seconds")
    all_parser.add_argument("--mip-gap", type=float, default=0.001, help="relative MIP gap tolerance")
    all_parser.add_argument("--quiet", action="store_true", help="suppress Gurobi console output")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(args.project_root) if args.project_root else find_project_root()

    if args.command == "prepare":
        result = build_realistic_inputs(project_root)
        print(json.dumps(result, indent=2))
        return

    if args.command == "solve":
        result = solve_realistic_model(
            project_root=project_root,
            cost_mode=args.cost_mode,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=0 if args.quiet else 1,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "all":
        prepare_result = build_realistic_inputs(project_root)
        solve_result = solve_realistic_model(
            project_root=project_root,
            cost_mode=args.cost_mode,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            output_flag=0 if args.quiet else 1,
        )
        print(json.dumps({"prepare": prepare_result, "solve": solve_result}, indent=2))


if __name__ == "__main__":
    main()

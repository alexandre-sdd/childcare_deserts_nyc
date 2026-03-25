from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "gurobipy is not available in this Python environment. "
        "Run this script with the Gurobi-enabled interpreter, for example "
        "`/opt/anaconda3/envs/gurobi/bin/python scripts/run_realistic_pipeline.py`."
    ) from exc


MIN_DIST_MILES = 0.06
SPECIAL_EQUIPMENT_COST = 100
TIME_LIMIT_SECONDS = 180
TARGET_MIP_GAP = 0.001


def find_project_root() -> Path:
    cwd = Path.cwd()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    return cwd


def zfill_zipcodes(df: pd.DataFrame) -> pd.DataFrame:
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    return df


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan

    radius_miles = 3958.7613

    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius_miles * c


def load_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ideal_dir = project_root / "data" / "processed" / "ideal"
    shared_dir = project_root / "data" / "processed" / "shared"

    zip_df = pd.read_csv(ideal_dir / "zipcode_demand_supply_ideal.csv")
    fac_clean = pd.read_csv(shared_dir / "clean_childcare_facilities.csv")
    candidate_geo = pd.read_csv(shared_dir / "potential_locations_geo_ready.csv")
    build_options = pd.read_csv(ideal_dir / "build_options_ideal.csv")

    return (
        zfill_zipcodes(zip_df),
        zfill_zipcodes(fac_clean),
        zfill_zipcodes(candidate_geo),
        build_options,
    )


def build_realistic_zip_inputs(zip_df: pd.DataFrame) -> pd.DataFrame:
    realistic_zip = zip_df.copy()

    zero_child_mask = realistic_zip["child_pop_0_12"].fillna(0) <= 0
    zero_under5_mask = realistic_zip["pop_0_5"].fillna(0) <= 0

    realistic_zip["zero_child_requirement_reset_flag"] = zero_child_mask.astype(int)
    realistic_zip["zero_under5_requirement_reset_flag"] = zero_under5_mask.astype(int)

    realistic_zip.loc[zero_child_mask, "total_threshold_rule"] = 0
    realistic_zip.loc[zero_child_mask, "req_total"] = 0
    realistic_zip.loc[zero_under5_mask, "req_under5"] = 0

    realistic_zip["gap_total"] = np.maximum(0, realistic_zip["req_total"] - realistic_zip["existing_total"])
    realistic_zip["gap_under5"] = np.maximum(0, realistic_zip["req_under5"] - realistic_zip["existing_under5"])

    realistic_zip["is_desert_total_before"] = np.where(
        zero_child_mask,
        0,
        (realistic_zip["existing_total"] <= realistic_zip["total_threshold_rule"]).astype(int),
    )
    realistic_zip["is_under5_short_before"] = np.where(
        zero_under5_mask,
        0,
        (realistic_zip["existing_under5"] < realistic_zip["req_under5"]).astype(int),
    )
    realistic_zip["needs_intervention_before"] = (
        (realistic_zip["gap_total"] > 0) | (realistic_zip["gap_under5"] > 0)
    ).astype(int)

    return realistic_zip.sort_values("zipcode").reset_index(drop=True)


def build_realistic_facility_inputs(fac_clean: pd.DataFrame) -> pd.DataFrame:
    fac_existing = fac_clean.copy()
    fac_existing = fac_existing[fac_existing["is_active_facility"] == 1].copy()
    fac_existing["facility_id"] = fac_existing["facility_id"].astype(str)
    fac_existing["n_f"] = pd.to_numeric(fac_existing["total_capacity"], errors="coerce").fillna(0)
    fac_existing["b_f"] = pd.to_numeric(fac_existing["under5_capacity_current"], errors="coerce").fillna(0)
    fac_existing["b_f"] = np.minimum(fac_existing["b_f"], fac_existing["n_f"])
    fac_existing = fac_existing[fac_existing["n_f"] > 0].copy()

    cut1 = np.floor(0.10 * fac_existing["n_f"]).astype(int)
    cut2 = np.floor(0.15 * fac_existing["n_f"]).astype(int)
    cut3 = np.floor(0.20 * fac_existing["n_f"]).astype(int)

    fac_existing["cut1"] = cut1
    fac_existing["cut2"] = cut2
    fac_existing["cut3"] = cut3
    fac_existing["cap1"] = cut1
    fac_existing["cap2"] = cut2 - cut1
    fac_existing["cap3"] = cut3 - cut2
    fac_existing["U_f_realistic"] = cut3

    fac_existing["coef1"] = (20000 + 200 * fac_existing["n_f"]) / fac_existing["n_f"]
    fac_existing["coef2"] = (20000 + 400 * fac_existing["n_f"]) / fac_existing["n_f"]
    fac_existing["coef3"] = (20000 + 1000 * fac_existing["n_f"]) / fac_existing["n_f"]

    keep_cols = [
        "facility_id",
        "facility_name",
        "program_type",
        "zipcode",
        "latitude",
        "longitude",
        "n_f",
        "b_f",
        "cut1",
        "cut2",
        "cut3",
        "cap1",
        "cap2",
        "cap3",
        "U_f_realistic",
        "coef1",
        "coef2",
        "coef3",
        "geo_missing_flag",
    ]
    return fac_existing[keep_cols].copy()


def build_distance_tables(
    candidate_geo: pd.DataFrame,
    facility_params: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_geo = candidate_geo.copy()
    candidate_geo["candidate_id"] = pd.to_numeric(candidate_geo["candidate_id"], errors="coerce").astype("Int64")

    candidate_required_cols = ["candidate_id", "zipcode", "latitude", "longitude"]
    missing_candidate_cols = [c for c in candidate_required_cols if c not in candidate_geo.columns]
    if missing_candidate_cols:
        raise KeyError(f"Missing candidate columns: {missing_candidate_cols}")

    cand_exist_pairs = candidate_geo.merge(
        facility_params[["facility_id", "zipcode", "latitude", "longitude"]],
        on="zipcode",
        how="inner",
        suffixes=("_cand", "_exist"),
    )
    cand_exist_pairs["distance_miles"] = cand_exist_pairs.apply(
        lambda row: haversine_miles(
            row["latitude_cand"],
            row["longitude_cand"],
            row["latitude_exist"],
            row["longitude_exist"],
        ),
        axis=1,
    )
    candidate_existing_conflicts = cand_exist_pairs[
        cand_exist_pairs["distance_miles"] < MIN_DIST_MILES
    ][["candidate_id", "facility_id", "zipcode", "distance_miles"]].drop_duplicates()

    cand1 = candidate_geo.rename(
        columns={"candidate_id": "candidate_id_1", "latitude": "latitude_1", "longitude": "longitude_1"}
    )
    cand2 = candidate_geo.rename(
        columns={"candidate_id": "candidate_id_2", "latitude": "latitude_2", "longitude": "longitude_2"}
    )
    candidate_candidate_pairs = cand1.merge(cand2, on="zipcode", how="inner")
    candidate_candidate_pairs = candidate_candidate_pairs[
        candidate_candidate_pairs["candidate_id_1"] < candidate_candidate_pairs["candidate_id_2"]
    ].copy()
    candidate_candidate_pairs["distance_miles"] = candidate_candidate_pairs.apply(
        lambda row: haversine_miles(
            row["latitude_1"],
            row["longitude_1"],
            row["latitude_2"],
            row["longitude_2"],
        ),
        axis=1,
    )
    candidate_candidate_conflicts = candidate_candidate_pairs[
        candidate_candidate_pairs["distance_miles"] < MIN_DIST_MILES
    ][["candidate_id_1", "candidate_id_2", "zipcode", "distance_miles"]].drop_duplicates()

    exist1 = facility_params.rename(
        columns={"facility_id": "facility_id_1", "latitude": "latitude_1", "longitude": "longitude_1"}
    )
    exist2 = facility_params.rename(
        columns={"facility_id": "facility_id_2", "latitude": "latitude_2", "longitude": "longitude_2"}
    )
    existing_existing_pairs = exist1.merge(exist2, on="zipcode", how="inner")
    existing_existing_pairs = existing_existing_pairs[
        existing_existing_pairs["facility_id_1"] < existing_existing_pairs["facility_id_2"]
    ].copy()
    existing_existing_pairs["distance_miles"] = existing_existing_pairs.apply(
        lambda row: haversine_miles(
            row["latitude_1"],
            row["longitude_1"],
            row["latitude_2"],
            row["longitude_2"],
        ),
        axis=1,
    )
    existing_existing_too_close = existing_existing_pairs[
        existing_existing_pairs["distance_miles"] < MIN_DIST_MILES
    ][["facility_id_1", "facility_id_2", "zipcode", "distance_miles"]].drop_duplicates()

    blocked_candidates = (
        candidate_existing_conflicts[["candidate_id"]].drop_duplicates().assign(blocked_by_existing=1)
    )
    candidate_geo_realistic = candidate_geo.merge(blocked_candidates, on="candidate_id", how="left")
    candidate_geo_realistic["blocked_by_existing"] = (
        candidate_geo_realistic["blocked_by_existing"].fillna(0).astype(int)
    )

    return (
        candidate_geo,
        candidate_geo_realistic,
        candidate_existing_conflicts,
        candidate_candidate_conflicts,
        existing_existing_too_close,
    )


def write_parameter_files(
    project_root: Path,
    realistic_zip: pd.DataFrame,
    facility_params: pd.DataFrame,
    candidate_geo_realistic: pd.DataFrame,
    candidate_build_options_feasible: pd.DataFrame,
    candidate_existing_conflicts: pd.DataFrame,
    candidate_candidate_conflicts: pd.DataFrame,
    existing_existing_too_close: pd.DataFrame,
) -> None:
    realistic_dir = project_root / "data" / "processed" / "realistic"
    realistic_dir.mkdir(parents=True, exist_ok=True)

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
                realistic_zip["zipcode"].nunique(),
                facility_params["facility_id"].nunique(),
                facility_params["n_f"].sum(),
                facility_params["b_f"].sum(),
                facility_params["U_f_realistic"].sum(),
                candidate_geo_realistic["candidate_id"].nunique(),
                candidate_geo_realistic["blocked_by_existing"].sum(),
                (candidate_geo_realistic["blocked_by_existing"] == 0).sum(),
                len(candidate_candidate_conflicts),
                len(candidate_existing_conflicts),
                len(existing_existing_too_close),
                realistic_zip["zero_child_requirement_reset_flag"].sum(),
                realistic_zip["zero_child_requirement_reset_flag"].sum(),
                facility_params["geo_missing_flag"].fillna(0).sum(),
            ],
        }
    )

    assumptions = {
        "distance_threshold_miles": MIN_DIST_MILES,
        "distance_enforced_within_zipcode_only": True,
        "existing_existing_conflicts_are_diagnostic_only": True,
        "candidate_existing_conflict_blocks_candidate": True,
        "use_under5_capacity_current_for_existing_capacity": True,
        "new_build_costs_and_capacities_reused_from_ideal_model": True,
        "realistic_expansion_cap_implemented_as_floor_20pct_total": True,
        "expansion_block_caps_from_cumulative_floors": True,
        "zero_child_zipcodes_have_zero_total_requirement": True,
        "cost_formulation_used_in_solver": "assignment_piecewise_total",
        "cost_formulation_superseded_from_previous_version": "additive_blocks",
    }

    realistic_zip.to_csv(realistic_dir / "zipcode_demand_supply_realistic.csv", index=False)
    facility_params.to_csv(realistic_dir / "facility_expansion_params_realistic.csv", index=False)
    candidate_geo_realistic.to_csv(realistic_dir / "candidate_sites_realistic.csv", index=False)
    candidate_build_options_feasible.to_csv(realistic_dir / "candidate_build_options_realistic.csv", index=False)
    candidate_existing_conflicts.to_csv(realistic_dir / "candidate_existing_conflicts.csv", index=False)
    candidate_candidate_conflicts.to_csv(realistic_dir / "candidate_candidate_conflicts.csv", index=False)
    existing_existing_too_close.to_csv(
        realistic_dir / "existing_existing_too_close_diagnostic.csv",
        index=False,
    )
    realistic_parameter_summary.to_csv(realistic_dir / "realistic_parameter_summary.csv", index=False)
    pd.DataFrame({"assumption": list(assumptions.keys()), "value": list(assumptions.values())}).to_csv(
        realistic_dir / "assumptions_realistic.csv",
        index=False,
    )
    with open(realistic_dir / "assumptions_realistic.json", "w", encoding="utf-8") as handle:
        json.dump(assumptions, handle, indent=2)


def apply_warm_start(
    project_root: Path,
    x_region1: gp.tupledict,
    x_region2: gp.tupledict,
    x_region3: gp.tupledict,
    z_region1: gp.tupledict,
    z_region2: gp.tupledict,
    z_region3: gp.tupledict,
    u: gp.tupledict,
    y: gp.tupledict,
    g: gp.tupledict,
    cut1: dict[str, int],
    cut2: dict[str, int],
    cut3: dict[str, int],
) -> None:
    solutions_dir = project_root / "results" / "realistic" / "solutions"
    facility_path = solutions_dir / "facility_solution_realistic.csv"
    build_path = solutions_dir / "new_build_solution_realistic.csv"

    if facility_path.exists():
        facility_start = pd.read_csv(facility_path)
        facility_start["facility_id"] = facility_start["facility_id"].astype(str)
        for _, row in facility_start.iterrows():
            facility_id = row["facility_id"]
            if facility_id not in cut3:
                continue

            expand_total = int(round(pd.to_numeric(row.get("expand_total", 0), errors="coerce") or 0))
            under5_slots = int(round(pd.to_numeric(row.get("u_under5", 0), errors="coerce") or 0))
            if expand_total <= 0:
                continue

            if expand_total <= cut1[facility_id]:
                x_region1[facility_id].Start = expand_total
                z_region1[facility_id].Start = 1
                x_region2[facility_id].Start = 0
                z_region2[facility_id].Start = 0
                x_region3[facility_id].Start = 0
                z_region3[facility_id].Start = 0
            elif expand_total <= cut2[facility_id]:
                x_region1[facility_id].Start = 0
                z_region1[facility_id].Start = 0
                x_region2[facility_id].Start = expand_total
                z_region2[facility_id].Start = 1
                x_region3[facility_id].Start = 0
                z_region3[facility_id].Start = 0
            elif expand_total <= cut3[facility_id]:
                x_region1[facility_id].Start = 0
                z_region1[facility_id].Start = 0
                x_region2[facility_id].Start = 0
                z_region2[facility_id].Start = 0
                x_region3[facility_id].Start = expand_total
                z_region3[facility_id].Start = 1
            u[facility_id].Start = min(under5_slots, expand_total)

    if build_path.exists():
        build_start = pd.read_csv(build_path)
        build_start["candidate_id"] = pd.to_numeric(build_start["candidate_id"], errors="coerce").astype("Int64")
        build_start["size"] = build_start["size"].astype(str)
        for _, row in build_start.iterrows():
            key = (int(row["candidate_id"]), row["size"])
            if key not in y:
                continue
            y[key].Start = 1
            g[key].Start = int(round(pd.to_numeric(row.get("new_under5_assigned", 0), errors="coerce") or 0))


def solve_realistic_model(project_root: Path) -> None:
    realistic_dir = project_root / "data" / "processed" / "realistic"
    results_dir = project_root / "results" / "realistic"
    log_dir = results_dir / "logs"
    solutions_dir = results_dir / "solutions"
    log_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    zip_df = pd.read_csv(realistic_dir / "zipcode_demand_supply_realistic.csv")
    fac_df = pd.read_csv(realistic_dir / "facility_expansion_params_realistic.csv")
    cand_df = pd.read_csv(realistic_dir / "candidate_build_options_realistic.csv")
    cc_conflict_df = pd.read_csv(realistic_dir / "candidate_candidate_conflicts.csv")

    zfill_zipcodes(zip_df)
    zfill_zipcodes(fac_df)
    zfill_zipcodes(cand_df)
    if not cc_conflict_df.empty:
        zfill_zipcodes(cc_conflict_df)

    zip_df["req_total"] = pd.to_numeric(zip_df["req_total"], errors="coerce").fillna(0).astype(int)
    zip_df["req_under5"] = pd.to_numeric(zip_df["req_under5"], errors="coerce").fillna(0).astype(int)
    zip_df["existing_total"] = pd.to_numeric(zip_df["existing_total"], errors="coerce").fillna(0).astype(int)
    zip_df["existing_under5"] = pd.to_numeric(zip_df["existing_under5"], errors="coerce").fillna(0).astype(int)

    fac_df["facility_id"] = fac_df["facility_id"].astype(str)
    for col in ["cut1", "cut2", "cut3", "cap1", "cap2", "cap3", "U_f_realistic"]:
        fac_df[col] = pd.to_numeric(fac_df[col], errors="coerce").fillna(0).astype(int)
    for col in ["n_f", "b_f", "coef1", "coef2", "coef3"]:
        fac_df[col] = pd.to_numeric(fac_df[col], errors="coerce").fillna(0)

    cand_df["candidate_id"] = pd.to_numeric(cand_df["candidate_id"], errors="coerce").astype(int)
    cand_df["size"] = cand_df["size"].astype(str)
    for col in ["new_total_capacity", "new_under5_capacity_max", "fixed_build_cost"]:
        cand_df[col] = pd.to_numeric(cand_df[col], errors="coerce").fillna(0).astype(int)

    if not cc_conflict_df.empty:
        cc_conflict_df["candidate_id_1"] = pd.to_numeric(
            cc_conflict_df["candidate_id_1"], errors="coerce"
        ).astype(int)
        cc_conflict_df["candidate_id_2"] = pd.to_numeric(
            cc_conflict_df["candidate_id_2"], errors="coerce"
        ).astype(int)

    zips = zip_df["zipcode"].tolist()
    facilities = fac_df["facility_id"].tolist()
    candidate_keys = list(zip(cand_df["candidate_id"], cand_df["size"]))
    candidate_sites = sorted(cand_df["candidate_id"].unique().tolist())

    req_total = dict(zip(zip_df["zipcode"], zip_df["req_total"]))
    req_under5 = dict(zip(zip_df["zipcode"], zip_df["req_under5"]))
    existing_total = dict(zip(zip_df["zipcode"], zip_df["existing_total"]))
    existing_under5 = dict(zip(zip_df["zipcode"], zip_df["existing_under5"]))

    fac_by_zip = fac_df.groupby("zipcode")["facility_id"].apply(list).to_dict()
    cut1 = dict(zip(fac_df["facility_id"], fac_df["cut1"]))
    cut2 = dict(zip(fac_df["facility_id"], fac_df["cut2"]))
    cut3 = dict(zip(fac_df["facility_id"], fac_df["cut3"]))
    coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
    coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
    coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

    build_total = {(row["candidate_id"], row["size"]): row["new_total_capacity"] for _, row in cand_df.iterrows()}
    build_under5_max = {
        (row["candidate_id"], row["size"]): row["new_under5_capacity_max"] for _, row in cand_df.iterrows()
    }
    build_cost = {(row["candidate_id"], row["size"]): row["fixed_build_cost"] for _, row in cand_df.iterrows()}
    key_to_zip = {(row["candidate_id"], row["size"]): row["zipcode"] for _, row in cand_df.iterrows()}
    sizes_by_candidate = cand_df.groupby("candidate_id")["size"].apply(list).to_dict()
    candsize_by_zip = cand_df.groupby("zipcode").apply(
        lambda group: list(zip(group["candidate_id"], group["size"]))
    ).to_dict()

    model = gp.Model("realistic_childcare_exact_piecewise")
    model.Params.LogFile = str(log_dir / "realistic_model.log")
    model.Params.MIPGap = TARGET_MIP_GAP
    model.Params.TimeLimit = TIME_LIMIT_SECONDS
    model.Params.MIPFocus = 1
    model.Params.OutputFlag = 1

    x_region1 = model.addVars(facilities, vtype=GRB.INTEGER, lb=0, name="x_region1")
    x_region2 = model.addVars(facilities, vtype=GRB.INTEGER, lb=0, name="x_region2")
    x_region3 = model.addVars(facilities, vtype=GRB.INTEGER, lb=0, name="x_region3")
    z_region1 = model.addVars(facilities, vtype=GRB.BINARY, name="z_region1")
    z_region2 = model.addVars(facilities, vtype=GRB.BINARY, name="z_region2")
    z_region3 = model.addVars(facilities, vtype=GRB.BINARY, name="z_region3")
    u = model.addVars(facilities, vtype=GRB.INTEGER, lb=0, name="u")

    y = model.addVars(candidate_keys, vtype=GRB.BINARY, name="y")
    g = model.addVars(candidate_keys, vtype=GRB.INTEGER, lb=0, name="g")

    for facility_id in facilities:
        model.addConstr(
            z_region1[facility_id] + z_region2[facility_id] + z_region3[facility_id] <= 1,
            name=f"one_region_{facility_id}",
        )

        model.addConstr(
            x_region1[facility_id] <= cut1[facility_id] * z_region1[facility_id],
            name=f"region1_ub_{facility_id}",
        )
        model.addConstr(
            x_region2[facility_id] <= cut2[facility_id] * z_region2[facility_id],
            name=f"region2_ub_{facility_id}",
        )
        model.addConstr(
            x_region3[facility_id] <= cut3[facility_id] * z_region3[facility_id],
            name=f"region3_ub_{facility_id}",
        )

        model.addConstr(
            x_region1[facility_id] >= z_region1[facility_id],
            name=f"region1_lb_{facility_id}",
        )
        model.addConstr(
            x_region2[facility_id] >= (cut1[facility_id] + 1) * z_region2[facility_id],
            name=f"region2_lb_{facility_id}",
        )
        model.addConstr(
            x_region3[facility_id] >= (cut2[facility_id] + 1) * z_region3[facility_id],
            name=f"region3_lb_{facility_id}",
        )

        model.addConstr(
            u[facility_id] <= x_region1[facility_id] + x_region2[facility_id] + x_region3[facility_id],
            name=f"under5_expand_cap_{facility_id}",
        )

    for candidate_id in candidate_sites:
        candidate_sizes = sizes_by_candidate.get(candidate_id, [])
        if candidate_sizes:
            model.addConstr(
                gp.quicksum(y[candidate_id, size] for size in candidate_sizes) <= 1,
                name=f"one_size_per_candidate_{candidate_id}",
            )

    for key in candidate_keys:
        model.addConstr(
            g[key] <= build_under5_max[key] * y[key],
            name=f"new_under5_cap_{key[0]}_{key[1]}",
        )

    if not cc_conflict_df.empty:
        for _, row in cc_conflict_df.iterrows():
            candidate_id_1 = int(row["candidate_id_1"])
            candidate_id_2 = int(row["candidate_id_2"])
            sizes_1 = sizes_by_candidate.get(candidate_id_1, [])
            sizes_2 = sizes_by_candidate.get(candidate_id_2, [])
            if not sizes_1 or not sizes_2:
                continue

            model.addConstr(
                gp.quicksum(y[candidate_id_1, size] for size in sizes_1)
                + gp.quicksum(y[candidate_id_2, size] for size in sizes_2)
                <= 1,
                name=f"cc_conflict_{candidate_id_1}_{candidate_id_2}",
            )

    for zipcode in zips:
        expand_total_expr = gp.quicksum(
            x_region1[facility_id] + x_region2[facility_id] + x_region3[facility_id]
            for facility_id in fac_by_zip.get(zipcode, [])
        )
        new_total_expr = gp.quicksum(
            build_total[key] * y[key] for key in candsize_by_zip.get(zipcode, [])
        )
        model.addConstr(
            existing_total[zipcode] + expand_total_expr + new_total_expr >= req_total[zipcode],
            name=f"total_req_{zipcode}",
        )

        expand_under5_expr = gp.quicksum(u[facility_id] for facility_id in fac_by_zip.get(zipcode, []))
        new_under5_expr = gp.quicksum(g[key] for key in candsize_by_zip.get(zipcode, []))
        model.addConstr(
            existing_under5[zipcode] + expand_under5_expr + new_under5_expr >= req_under5[zipcode],
            name=f"under5_req_{zipcode}",
        )

    expansion_cost_expr = gp.quicksum(
        coef1[facility_id] * x_region1[facility_id]
        + coef2[facility_id] * x_region2[facility_id]
        + coef3[facility_id] * x_region3[facility_id]
        for facility_id in facilities
    )
    new_build_cost_expr = gp.quicksum(build_cost[key] * y[key] for key in candidate_keys)
    under5_equipment_cost_expr = (
        gp.quicksum(SPECIAL_EQUIPMENT_COST * u[facility_id] for facility_id in facilities)
        + gp.quicksum(SPECIAL_EQUIPMENT_COST * g[key] for key in candidate_keys)
    )

    model.setObjective(
        expansion_cost_expr + new_build_cost_expr + under5_equipment_cost_expr,
        GRB.MINIMIZE,
    )

    apply_warm_start(
        project_root,
        x_region1,
        x_region2,
        x_region3,
        z_region1,
        z_region2,
        z_region3,
        u,
        y,
        g,
        cut1,
        cut2,
        cut3,
    )

    start_time = time.time()
    model.optimize()
    solve_seconds = time.time() - start_time

    has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0
    if not has_solution:
        raise RuntimeError(f"Realistic model did not produce a usable solution. Status={model.Status}")

    facility_solution = fac_df.copy()
    facility_solution["x_region1"] = facility_solution["facility_id"].map(lambda f: x_region1[f].X)
    facility_solution["x_region2"] = facility_solution["facility_id"].map(lambda f: x_region2[f].X)
    facility_solution["x_region3"] = facility_solution["facility_id"].map(lambda f: x_region3[f].X)
    facility_solution["u_under5"] = facility_solution["facility_id"].map(lambda f: u[f].X)
    facility_solution["expand_total"] = (
        facility_solution["x_region1"] + facility_solution["x_region2"] + facility_solution["x_region3"]
    )
    facility_solution["selected_region"] = np.select(
        [
            facility_solution["x_region1"] > 0,
            facility_solution["x_region2"] > 0,
            facility_solution["x_region3"] > 0,
        ],
        ["0_10_pct", "10_15_pct", "15_20_pct"],
        default="none",
    )
    facility_solution["expansion_cost"] = (
        facility_solution["coef1"] * facility_solution["x_region1"]
        + facility_solution["coef2"] * facility_solution["x_region2"]
        + facility_solution["coef3"] * facility_solution["x_region3"]
    )
    facility_solution["under5_equipment_cost_expansion"] = SPECIAL_EQUIPMENT_COST * facility_solution["u_under5"]
    facility_solution["total_expansion_related_cost"] = (
        facility_solution["expansion_cost"] + facility_solution["under5_equipment_cost_expansion"]
    )
    facility_solution = facility_solution[facility_solution["expand_total"] > 1e-6].copy()

    build_rows: list[dict[str, object]] = []
    for key in candidate_keys:
        if y[key].X <= 0.5:
            continue
        candidate_id, size = key
        row = cand_df[(cand_df["candidate_id"] == candidate_id) & (cand_df["size"] == size)].iloc[0].to_dict()
        row["build_selected"] = int(round(y[key].X))
        row["new_under5_assigned"] = g[key].X
        row["under5_equipment_cost_newbuild"] = SPECIAL_EQUIPMENT_COST * g[key].X
        row["total_newbuild_related_cost"] = row["fixed_build_cost"] + row["under5_equipment_cost_newbuild"]
        build_rows.append(row)
    new_build_solution = pd.DataFrame(build_rows)

    if facility_solution.empty:
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
    else:
        facility_expand_zip = facility_solution.groupby("zipcode", as_index=False).agg(
            expanded_total=("expand_total", "sum"),
            expanded_under5=("u_under5", "sum"),
            expansion_cost=("expansion_cost", "sum"),
            under5_equipment_cost_expansion=("under5_equipment_cost_expansion", "sum"),
            total_expansion_related_cost=("total_expansion_related_cost", "sum"),
        )

    if new_build_solution.empty:
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
    else:
        new_build_zip = new_build_solution.groupby("zipcode", as_index=False).agg(
            n_new_facilities=("candidate_id", "count"),
            new_total_capacity=("new_total_capacity", "sum"),
            new_under5_capacity=("new_under5_assigned", "sum"),
            new_build_cost=("fixed_build_cost", "sum"),
            under5_equipment_cost_newbuild=("under5_equipment_cost_newbuild", "sum"),
            total_newbuild_related_cost=("total_newbuild_related_cost", "sum"),
        )

    zipcode_solution = zip_df.copy()
    zipcode_solution = zipcode_solution.merge(facility_expand_zip, on="zipcode", how="left")
    zipcode_solution = zipcode_solution.merge(new_build_zip, on="zipcode", how="left")

    fill_zero_cols = [
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
    ]
    for col in fill_zero_cols:
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
        (zipcode_solution["total_slack"] >= -1e-6) & (zipcode_solution["under5_slack"] >= -1e-6)
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

    run_metadata = pd.DataFrame(
        {
            "metric": [
                "model_status",
                "objective_value",
                "mip_gap",
                "solve_seconds",
                "num_variables",
                "num_constraints",
                "solution_count",
            ],
            "value": [
                model.Status,
                model.ObjVal,
                model.MIPGap if hasattr(model, "MIPGap") else np.nan,
                solve_seconds,
                model.NumVars,
                model.NumConstrs,
                model.SolCount,
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
                "zipcodes_meeting_all_requirements_after",
            ],
            "value": [
                len(facility_solution),
                facility_solution["expand_total"].sum() if not facility_solution.empty else 0,
                facility_solution["u_under5"].sum() if not facility_solution.empty else 0,
                len(new_build_solution) if not new_build_solution.empty else 0,
                new_build_solution["new_total_capacity"].sum() if not new_build_solution.empty else 0,
                new_build_solution["new_under5_assigned"].sum() if not new_build_solution.empty else 0,
                zipcode_solution["all_requirements_met_after"].sum(),
            ],
        }
    )

    facility_solution.to_csv(solutions_dir / "facility_solution_realistic.csv", index=False)
    zipcode_solution.to_csv(solutions_dir / "zipcode_solution_realistic.csv", index=False)
    objective_breakdown.to_csv(solutions_dir / "objective_breakdown_realistic.csv", index=False)
    run_metadata.to_csv(solutions_dir / "run_metadata_realistic.csv", index=False)
    summary_stats.to_csv(solutions_dir / "summary_stats_realistic.csv", index=False)

    if new_build_solution.empty:
        pd.DataFrame(
            columns=cand_df.columns.tolist()
            + [
                "build_selected",
                "new_under5_assigned",
                "under5_equipment_cost_newbuild",
                "total_newbuild_related_cost",
            ]
        ).to_csv(solutions_dir / "new_build_solution_realistic.csv", index=False)
    else:
        new_build_solution.to_csv(solutions_dir / "new_build_solution_realistic.csv", index=False)


def main() -> None:
    project_root = find_project_root()
    zip_df, fac_clean, candidate_geo, build_options = load_inputs(project_root)

    realistic_zip = build_realistic_zip_inputs(zip_df)
    facility_params = build_realistic_facility_inputs(fac_clean)

    (
        candidate_geo,
        candidate_geo_realistic,
        candidate_existing_conflicts,
        candidate_candidate_conflicts,
        existing_existing_too_close,
    ) = build_distance_tables(candidate_geo, facility_params)

    candidate_build_options_realistic = candidate_geo.merge(build_options, how="cross")
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

    write_parameter_files(
        project_root,
        realistic_zip,
        facility_params,
        candidate_geo_realistic,
        candidate_build_options_feasible,
        candidate_existing_conflicts,
        candidate_candidate_conflicts,
        existing_existing_too_close,
    )
    solve_realistic_model(project_root)


if __name__ == "__main__":
    main()

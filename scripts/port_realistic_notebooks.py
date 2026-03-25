#!/usr/bin/env python3
"""Rebuild the Part 2 notebooks so they become the notebook source of truth."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "gurobi",
        "language": "python",
        "name": "gurobi",
    },
    "language_info": {
        "name": "python",
        "version": "3.12.12",
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook_05():
    cells = [
        md(
            """
            # 05_parameter_estimation_realistic

            This notebook converts the shared cleaned datasets and the Part 1 demand table
            into model-ready inputs for the realistic childcare expansion model.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            import numpy as np
            import pandas as pd
            """
        ),
        code(
            """
            # Display options
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 180)
            pd.set_option("display.max_rows", 200)
            """
        ),
        code(
            """
            # Project paths
            PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

            SHARED_DIR = PROJECT_ROOT / "data" / "processed" / "shared"
            IDEAL_DIR = PROJECT_ROOT / "data" / "processed" / "ideal"
            REALISTIC_DIR = PROJECT_ROOT / "data" / "processed" / "realistic"

            REALISTIC_DIR.mkdir(parents=True, exist_ok=True)

            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("SHARED_DIR:", SHARED_DIR)
            print("IDEAL_DIR:", IDEAL_DIR)
            print("REALISTIC_DIR:", REALISTIC_DIR)
            """
        ),
        code(
            """
            # Check required input files
            required_files = [
                "clean_childcare_facilities.csv",
                "facility_geo_ready.csv",
                "potential_locations_geo_ready.csv",
                "zipcode_demand_supply_ideal.csv",
                "build_options_ideal.csv",
            ]

            for file_name in required_files:
                base_dir = SHARED_DIR if file_name.endswith(".csv") and "ideal" not in file_name and "build_options" not in file_name else IDEAL_DIR
                if file_name in {"zipcode_demand_supply_ideal.csv", "build_options_ideal.csv"}:
                    path = IDEAL_DIR / file_name
                else:
                    path = SHARED_DIR / file_name
                print(file_name, "exists ->", path.exists())
            """
        ),
        code(
            """
            # Load input tables
            fac_clean = pd.read_csv(SHARED_DIR / "clean_childcare_facilities.csv")
            facility_geo = pd.read_csv(SHARED_DIR / "facility_geo_ready.csv")
            candidate_geo = pd.read_csv(SHARED_DIR / "potential_locations_geo_ready.csv")
            zip_df_ideal = pd.read_csv(IDEAL_DIR / "zipcode_demand_supply_ideal.csv")
            build_options = pd.read_csv(IDEAL_DIR / "build_options_ideal.csv")
            """
        ),
        code(
            """
            # Quick preview
            datasets = {
                "fac_clean": fac_clean,
                "facility_geo": facility_geo,
                "candidate_geo": candidate_geo,
                "zip_df_ideal": zip_df_ideal,
                "build_options": build_options,
            }

            for name, df in datasets.items():
                print(f"\\n{name}: {df.shape}")
                display(df.head())
            """
        ),
        code(
            """
            # Realistic-model assumptions
            REALISTIC_ASSUMPTIONS = {
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
            """
        ),
        code(
            """
            # Helper functions
            def standardize_zipcode(series):
                return (
                    series.astype(str)
                    .str.strip()
                    .str.replace(r"\\.0$", "", regex=True)
                    .str.zfill(5)
                )


            def haversine_miles(lat1, lon1, lat2, lon2):
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
            """
        ),
        md(
            """
            ## Realistic Expansion Parameters

            This section keeps the Part 1 facility universe but changes the realistic
            expansion cap logic to preserve the intended total `floor(0.20 * n_f)` limit.
            """
        ),
        code(
            """
            # Standardize keys and keep active facilities with positive capacity
            for df in [fac_clean, facility_geo, candidate_geo, zip_df_ideal]:
                if "zipcode" in df.columns:
                    df["zipcode"] = standardize_zipcode(df["zipcode"])

            fac_clean["facility_id"] = fac_clean["facility_id"].astype(str)
            facility_geo["facility_id"] = facility_geo["facility_id"].astype(str)
            candidate_geo["candidate_id"] = pd.to_numeric(candidate_geo["candidate_id"], errors="coerce").astype("Int64")

            fac_existing = fac_clean.copy()
            fac_existing = fac_existing[fac_existing["is_active_facility"] == 1].copy()
            fac_existing = fac_existing[fac_existing["total_capacity"].fillna(0) > 0].copy()

            fac_existing["n_f"] = pd.to_numeric(fac_existing["total_capacity"], errors="coerce").fillna(0).astype(int)
            fac_existing["b_f"] = pd.to_numeric(fac_existing["under5_capacity_current"], errors="coerce").fillna(0).astype(int)

            fac_existing = fac_existing.merge(
                facility_geo[["facility_id", "latitude", "longitude", "zipcode"]],
                on="facility_id",
                how="left",
                suffixes=("", "_geo"),
            )
            fac_existing["zipcode"] = fac_existing["zipcode"].fillna(fac_existing["zipcode_geo"])
            fac_existing = fac_existing.drop(columns=["zipcode_geo"], errors="ignore")

            display(
                fac_existing[[
                    "facility_id", "facility_name", "zipcode", "n_f", "b_f",
                    "latitude", "longitude"
                ]].head()
            )
            """
        ),
        code(
            """
            # Realistic expansion cap design
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

            facility_expansion_params_realistic = fac_existing[[
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
            ]].copy()

            display(facility_expansion_params_realistic.head())
            """
        ),
        code(
            """
            # Facility parameter sanity checks
            print("Any negative block capacities?", (facility_expansion_params_realistic[["cap1", "cap2", "cap3"]] < 0).any().any())
            print("Any missing cost coefficients?", facility_expansion_params_realistic[["coef1", "coef2", "coef3"]].isna().any().any())
            print("Citywide realistic expansion capacity:", int(facility_expansion_params_realistic["U_f_realistic"].sum()))
            print("Existing facilities missing coordinates:", int(facility_expansion_params_realistic["latitude"].isna().sum()))
            """
        ),
        md(
            """
            ## Demand Repair

            Part 2 keeps the Part 1 demand construction except for the zero-child ZIP code
            repair, which removes artificial total-slot requirements where observed child
            population is zero.
            """
        ),
        code(
            """
            # Start from the Part 1 demand table and repair zero-child ZIP codes
            zipcode_demand_supply_realistic = zip_df_ideal.copy()
            zipcode_demand_supply_realistic["zipcode"] = standardize_zipcode(zipcode_demand_supply_realistic["zipcode"])

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

            print("ZIP codes with zero child population:", int(zero_child_mask.sum()))
            print("ZIP codes with req_total reset to zero:", zero_child_req_total_fixed)
            display(
                zipcode_demand_supply_realistic.loc[
                    zero_child_mask,
                    ["zipcode", "child_pop_0_12", "req_total", "existing_total", "gap_total"]
                ].head(15)
            )
            """
        ),
        md(
            """
            ## Candidate-Site Screening

            This section builds realistic candidate-site options and the distance-based
            conflict tables used in the optimization model.
            """
        ),
        code(
            """
            # Candidate-site build options
            candidate_geo = candidate_geo.copy()
            candidate_geo["zipcode"] = standardize_zipcode(candidate_geo["zipcode"])

            candidate_build_options_realistic = (
                candidate_geo.assign(key=1)
                .merge(build_options.assign(key=1), on="key", how="inner")
                .drop(columns=["key"])
            )

            print("Candidate sites:", candidate_geo.shape)
            print("Candidate-size options:", candidate_build_options_realistic.shape)
            """
        ),
        code(
            """
            # Candidate-existing conflict table
            MIN_DIST = REALISTIC_ASSUMPTIONS["distance_threshold_miles"]

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
                cand_exist_pairs["distance_miles"] < MIN_DIST
            ][["candidate_id", "facility_id", "zipcode", "distance_miles"]].drop_duplicates()

            print("Candidate-existing conflicts:", candidate_existing_conflicts.shape)
            display(candidate_existing_conflicts.head())
            """
        ),
        code(
            """
            # Candidate-candidate conflict table
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
                candidate_candidate_pairs["distance_miles"] < MIN_DIST
            ][["candidate_id_1", "candidate_id_2", "zipcode", "distance_miles"]].drop_duplicates()

            print("Candidate-candidate conflicts:", candidate_candidate_conflicts.shape)
            display(candidate_candidate_conflicts.head())
            """
        ),
        code(
            """
            # Existing-existing close pairs (diagnostic only)
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
                existing_existing_pairs["distance_miles"] < MIN_DIST
            ][["facility_id_1", "facility_id_2", "zipcode", "distance_miles"]].drop_duplicates()

            print("Existing-existing close pairs (diagnostic):", existing_existing_too_close.shape)
            display(existing_existing_too_close.head())
            """
        ),
        code(
            """
            # Blocked candidate flags and feasible candidate-size options
            blocked_candidates = (
                candidate_existing_conflicts[["candidate_id"]]
                .drop_duplicates()
                .assign(blocked_by_existing=1)
            )

            candidate_geo_realistic = candidate_geo.merge(
                blocked_candidates,
                on="candidate_id",
                how="left",
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

            print("Blocked candidate sites:", int(candidate_geo_realistic["blocked_by_existing"].sum()))
            print("Feasible candidate sites:", int((candidate_geo_realistic["blocked_by_existing"] == 0).sum()))
            print("Feasible candidate-size options:", candidate_build_options_feasible.shape)
            """
        ),
        code(
            """
            # Summary table for the report
            realistic_parameter_summary = pd.DataFrame({
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
            })

            display(realistic_parameter_summary)
            """
        ),
        md(
            """
            ## Export

            The exported Part 2 parameter files are the direct inputs for the realistic
            optimization notebook.
            """
        ),
        code(
            """
            # Save assumptions table
            with open(REALISTIC_DIR / "assumptions_realistic.json", "w", encoding="utf-8") as f:
                json.dump(REALISTIC_ASSUMPTIONS, f, ensure_ascii=False, indent=2)

            assumptions_realistic = pd.DataFrame(
                {"assumption": list(REALISTIC_ASSUMPTIONS.keys()), "value": list(REALISTIC_ASSUMPTIONS.values())}
            )

            assumptions_realistic
            """
        ),
        code(
            """
            # Export all realistic parameter files
            facility_expansion_params_realistic.to_csv(
                REALISTIC_DIR / "facility_expansion_params_realistic.csv",
                index=False,
            )
            zipcode_demand_supply_realistic.to_csv(
                REALISTIC_DIR / "zipcode_demand_supply_realistic.csv",
                index=False,
            )
            candidate_geo_realistic.to_csv(
                REALISTIC_DIR / "candidate_sites_realistic.csv",
                index=False,
            )
            candidate_build_options_feasible.to_csv(
                REALISTIC_DIR / "candidate_build_options_realistic.csv",
                index=False,
            )
            candidate_existing_conflicts.to_csv(
                REALISTIC_DIR / "candidate_existing_conflicts.csv",
                index=False,
            )
            candidate_candidate_conflicts.to_csv(
                REALISTIC_DIR / "candidate_candidate_conflicts.csv",
                index=False,
            )
            existing_existing_too_close.to_csv(
                REALISTIC_DIR / "existing_existing_too_close_diagnostic.csv",
                index=False,
            )
            realistic_parameter_summary.to_csv(
                REALISTIC_DIR / "realistic_parameter_summary.csv",
                index=False,
            )
            assumptions_realistic.to_csv(
                REALISTIC_DIR / "assumptions_realistic.csv",
                index=False,
            )

            print("Realistic parameter files saved to:", REALISTIC_DIR)
            """
        ),
        code(
            """
            # Final QA preview
            print("facility_expansion_params_realistic:", facility_expansion_params_realistic.shape)
            print("zipcode_demand_supply_realistic:", zipcode_demand_supply_realistic.shape)
            print("candidate_sites_realistic:", candidate_geo_realistic.shape)
            print("candidate_build_options_feasible:", candidate_build_options_feasible.shape)
            print("candidate_existing_conflicts:", candidate_existing_conflicts.shape)
            print("candidate_candidate_conflicts:", candidate_candidate_conflicts.shape)
            print("existing_existing_too_close:", existing_existing_too_close.shape)
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = NOTEBOOK_METADATA
    return nb


def build_notebook_06():
    cells = [
        md(
            """
            # 06_realistic_model

            This notebook builds and solves the realistic childcare expansion model using
            the Part 2 inputs produced by `05_parameter_estimation_realistic.ipynb`.
            """
        ),
        code(
            """
            import time
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import gurobipy as gp
            from gurobipy import GRB
            """
        ),
        code(
            """
            # Display options
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.width", 180)
            pd.set_option("display.max_rows", 200)
            """
        ),
        code(
            """
            # Project paths
            PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

            REALISTIC_DIR = PROJECT_ROOT / "data" / "processed" / "realistic"
            RESULTS_DIR = PROJECT_ROOT / "results" / "realistic"
            SOLUTIONS_DIR = RESULTS_DIR / "solutions"
            LOG_DIR = RESULTS_DIR / "logs"

            SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("REALISTIC_DIR:", REALISTIC_DIR)
            print("SOLUTIONS_DIR:", SOLUTIONS_DIR)
            """
        ),
        code(
            """
            # Check required input files
            required_files = [
                "zipcode_demand_supply_realistic.csv",
                "facility_expansion_params_realistic.csv",
                "candidate_build_options_realistic.csv",
                "candidate_candidate_conflicts.csv",
            ]

            for file_name in required_files:
                path = REALISTIC_DIR / file_name
                print(file_name, "exists ->", path.exists())
            """
        ),
        code(
            """
            # Load realistic model inputs
            zip_df = pd.read_csv(REALISTIC_DIR / "zipcode_demand_supply_realistic.csv")
            fac_df = pd.read_csv(REALISTIC_DIR / "facility_expansion_params_realistic.csv")
            cand_df = pd.read_csv(REALISTIC_DIR / "candidate_build_options_realistic.csv")
            cc_conflict_df = pd.read_csv(REALISTIC_DIR / "candidate_candidate_conflicts.csv")
            """
        ),
        code(
            """
            # Basic preview
            print("zipcode_demand_supply_realistic:", zip_df.shape)
            display(zip_df.head())

            print("facility_expansion_params_realistic:", fac_df.shape)
            display(fac_df.head())

            print("candidate_build_options_realistic:", cand_df.shape)
            display(cand_df.head())
            """
        ),
        code(
            """
            # Solve-mode assumptions
            COST_MODE = "additive_blocks"  # or "assignment_piecewise_total"
            SPECIAL_EQUIPMENT_COST = 100
            """
        ),
        code(
            """
            # Final type cleanup before modeling
            def standardize_zipcode(series):
                return (
                    series.astype(str)
                    .str.strip()
                    .str.replace(r"\\.0$", "", regex=True)
                    .str.zfill(5)
                )


            zip_df["zipcode"] = standardize_zipcode(zip_df["zipcode"])
            fac_df["zipcode"] = standardize_zipcode(fac_df["zipcode"])
            cand_df["zipcode"] = standardize_zipcode(cand_df["zipcode"])

            fac_df["facility_id"] = fac_df["facility_id"].astype(str)
            cand_df["candidate_id"] = pd.to_numeric(cand_df["candidate_id"], errors="coerce").astype(int)
            cand_df["size"] = cand_df["size"].astype(str)

            if not cc_conflict_df.empty:
                cc_conflict_df["zipcode"] = standardize_zipcode(cc_conflict_df["zipcode"])
                cc_conflict_df["candidate_id_1"] = pd.to_numeric(cc_conflict_df["candidate_id_1"], errors="coerce").astype(int)
                cc_conflict_df["candidate_id_2"] = pd.to_numeric(cc_conflict_df["candidate_id_2"], errors="coerce").astype(int)
            """
        ),
        code(
            """
            # Build sets and parameter dictionaries
            Z = zip_df["zipcode"].tolist()
            F = fac_df["facility_id"].tolist()
            candidate_sites = sorted(cand_df["candidate_id"].unique().tolist())
            cand_key_tuples = list(zip(cand_df["candidate_id"], cand_df["size"]))

            req_total = dict(zip(zip_df["zipcode"], zip_df["req_total"]))
            req_under5 = dict(zip(zip_df["zipcode"], zip_df["req_under5"]))
            existing_total = dict(zip(zip_df["zipcode"], zip_df["existing_total"]))
            existing_under5 = dict(zip(zip_df["zipcode"], zip_df["existing_under5"]))

            fac_by_zip = fac_df.groupby("zipcode")["facility_id"].apply(list).to_dict()
            candsize_by_zip = (
                cand_df.groupby("zipcode")[["candidate_id", "size"]]
                .apply(lambda g: list(map(tuple, g.to_numpy())))
                .to_dict()
            )
            sizes_by_candidate = cand_df.groupby("candidate_id")["size"].apply(list).to_dict()

            build_total = {(row["candidate_id"], row["size"]): row["new_total_capacity"] for _, row in cand_df.iterrows()}
            build_under5_max = {(row["candidate_id"], row["size"]): row["new_under5_capacity_max"] for _, row in cand_df.iterrows()}
            build_cost = {(row["candidate_id"], row["size"]): row["fixed_build_cost"] for _, row in cand_df.iterrows()}

            print("Number of zipcodes:", len(Z))
            print("Number of facilities:", len(F))
            print("Number of candidate sites:", len(candidate_sites))
            print("Number of candidate-size options:", len(cand_key_tuples))
            """
        ),
        code(
            """
            # Validation checks
            print("Any zipcode missing req_total?", int(zip_df["req_total"].isna().sum()))
            print("Any zipcode missing req_under5?", int(zip_df["req_under5"].isna().sum()))
            print("Any negative realistic block capacities?", (fac_df[["cap1", "cap2", "cap3"]] < 0).any().any())
            print("Any blocked candidate rows still present?", int(cand_df["blocked_by_existing"].fillna(0).sum()) if "blocked_by_existing" in cand_df.columns else 0)
            """
        ),
        md(
            """
            ## Model Design

            The default formulation uses additive expansion blocks because it solves cleanly
            at city scale. An optional assignment-style piecewise-total mode is included for
            comparison, but it is slower.
            """
        ),
        code(
            """
            # Solver parameters and helper builders
            log_file = LOG_DIR / "realistic_model.log"


            def build_additive_expansion_artifacts(model, fac_df):
                cap1 = dict(zip(fac_df["facility_id"], fac_df["cap1"]))
                cap2 = dict(zip(fac_df["facility_id"], fac_df["cap2"]))
                cap3 = dict(zip(fac_df["facility_id"], fac_df["cap3"]))
                coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
                coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
                coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

                x_low = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x_low")
                x_mid = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x_mid")
                x_high = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x_high")
                u = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="u")

                for facility_id in F:
                    model.addConstr(x_low[facility_id] <= cap1[facility_id], name=f"cap1_{facility_id}")
                    model.addConstr(x_mid[facility_id] <= cap2[facility_id], name=f"cap2_{facility_id}")
                    model.addConstr(x_high[facility_id] <= cap3[facility_id], name=f"cap3_{facility_id}")
                    model.addConstr(
                        u[facility_id] <= x_low[facility_id] + x_mid[facility_id] + x_high[facility_id],
                        name=f"under5_expand_cap_{facility_id}",
                    )

                expansion_cost_expr = gp.quicksum(
                    coef1[facility_id] * x_low[facility_id]
                    + coef2[facility_id] * x_mid[facility_id]
                    + coef3[facility_id] * x_high[facility_id]
                    for facility_id in F
                )

                expand_total_expr = {
                    facility_id: x_low[facility_id] + x_mid[facility_id] + x_high[facility_id]
                    for facility_id in F
                }

                return {
                    "mode": "additive_blocks",
                    "u": u,
                    "x_low": x_low,
                    "x_mid": x_mid,
                    "x_high": x_high,
                    "expand_total_expr": expand_total_expr,
                    "expansion_cost_expr": expansion_cost_expr,
                }


            def build_assignment_piecewise_artifacts(model, fac_df):
                cap10 = dict(zip(fac_df["facility_id"], fac_df["cap10_cum"]))
                cap15 = dict(zip(fac_df["facility_id"], fac_df["cap15_cum"]))
                cap20 = dict(zip(fac_df["facility_id"], fac_df["cap20_cum"]))
                coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
                coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
                coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

                x_total = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x_total")
                u = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="u")
                z_low = model.addVars(F, vtype=GRB.BINARY, name="z_low")
                z_mid = model.addVars(F, vtype=GRB.BINARY, name="z_mid")
                z_high = model.addVars(F, vtype=GRB.BINARY, name="z_high")
                expansion_cost = model.addVars(F, lb=0.0, name="expansion_cost")

                for facility_id in F:
                    model.addConstr(z_low[facility_id] + z_mid[facility_id] + z_high[facility_id] <= 1, name=f"one_regime_{facility_id}")
                    model.addConstr(
                        x_total[facility_id]
                        <= cap10[facility_id] * z_low[facility_id]
                        + cap15[facility_id] * z_mid[facility_id]
                        + cap20[facility_id] * z_high[facility_id],
                        name=f"regime_upper_{facility_id}",
                    )
                    model.addConstr(
                        x_total[facility_id]
                        >= z_low[facility_id]
                        + (cap10[facility_id] + 1) * z_mid[facility_id]
                        + (cap15[facility_id] + 1) * z_high[facility_id],
                        name=f"regime_lower_{facility_id}",
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
                    "u": u,
                    "x_total": x_total,
                    "z_low": z_low,
                    "z_mid": z_mid,
                    "z_high": z_high,
                    "expansion_cost_var": expansion_cost,
                    "expand_total_expr": x_total,
                    "expansion_cost_expr": gp.quicksum(expansion_cost[facility_id] for facility_id in F),
                }
            """
        ),
        code(
            """
            # Build and solve the realistic MILP
            model = gp.Model("realistic_childcare")
            model.Params.LogFile = str(log_file)
            model.Params.MIPGap = 0.001
            model.Params.TimeLimit = 600
            model.Params.OutputFlag = 1
            model.Params.Presolve = 2
            model.Params.Threads = 0

            y = model.addVars(cand_key_tuples, vtype=GRB.BINARY, name="y")
            g = model.addVars(cand_key_tuples, vtype=GRB.INTEGER, lb=0, name="g")

            if COST_MODE == "additive_blocks":
                model_artifacts = build_additive_expansion_artifacts(model, fac_df)
            elif COST_MODE == "assignment_piecewise_total":
                model_artifacts = build_assignment_piecewise_artifacts(model, fac_df)
            else:
                raise ValueError(f"Unsupported COST_MODE: {COST_MODE}")

            expand_total_expr = model_artifacts["expand_total_expr"]
            u = model_artifacts["u"]
            expansion_cost_expr = model_artifacts["expansion_cost_expr"]
            """
        ),
        code(
            """
            # Candidate-site constraints
            for candidate_id in candidate_sites:
                candidate_sizes = sizes_by_candidate.get(candidate_id, [])
                if candidate_sizes:
                    model.addConstr(
                        gp.quicksum(y[candidate_id, size] for size in candidate_sizes) <= 1,
                        name=f"one_size_per_candidate_{candidate_id}",
                    )

            for candidate_id, size in cand_key_tuples:
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
            """
        ),
        code(
            """
            # Zipcode-level service constraints
            for zipcode in Z:
                model.addConstr(
                    existing_total[zipcode]
                    + gp.quicksum(expand_total_expr[facility_id] for facility_id in fac_by_zip.get(zipcode, []))
                    + gp.quicksum(
                        build_total[(candidate_id, size)] * y[candidate_id, size]
                        for candidate_id, size in candsize_by_zip.get(zipcode, [])
                    )
                    >= req_total[zipcode],
                    name=f"total_req_{zipcode}",
                )

            for zipcode in Z:
                model.addConstr(
                    existing_under5[zipcode]
                    + gp.quicksum(u[facility_id] for facility_id in fac_by_zip.get(zipcode, []))
                    + gp.quicksum(g[candidate_id, size] for candidate_id, size in candsize_by_zip.get(zipcode, []))
                    >= req_under5[zipcode],
                    name=f"under5_req_{zipcode}",
                )
            """
        ),
        md(
            """
            ## Objective Function

            The realistic model minimizes expansion cost, fixed build cost, and the under-5
            equipment surcharge.
            """
        ),
        code(
            """
            # Set objective and optimize
            new_build_cost_expr = gp.quicksum(
                build_cost[(candidate_id, size)] * y[candidate_id, size]
                for candidate_id, size in cand_key_tuples
            )

            under5_equipment_cost_expr = (
                gp.quicksum(SPECIAL_EQUIPMENT_COST * u[facility_id] for facility_id in F)
                + gp.quicksum(SPECIAL_EQUIPMENT_COST * g[candidate_id, size] for candidate_id, size in cand_key_tuples)
            )

            model.setObjective(
                expansion_cost_expr + new_build_cost_expr + under5_equipment_cost_expr,
                GRB.MINIMIZE,
            )

            model.update()
            print("Number of variables:", model.NumVars)
            print("Number of constraints:", model.NumConstrs)

            start_time = time.time()
            model.optimize()
            solve_seconds = time.time() - start_time

            print("Model status code:", model.Status)
            print(f"Solve time: {solve_seconds:.2f} seconds")
            """
        ),
        code(
            """
            # Interpret solve status and optionally diagnose infeasibility
            status_map = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.SUBOPTIMAL: "SUBOPTIMAL",
                GRB.INTERRUPTED: "INTERRUPTED",
            }

            if model.Status == GRB.INFEASIBLE:
                print("Model is infeasible. Computing IIS...")
                model.computeIIS()
                iis_path = SOLUTIONS_DIR / "realistic_model.ilp"
                model.write(str(iis_path))
                print("IIS written to:", iis_path)

            has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0
            print("Has solution:", has_solution)
            print("Status text:", status_map.get(model.Status, str(model.Status)))
            """
        ),
        code(
            """
            # Facility expansion solution table
            if has_solution:
                facility_solution = fac_df.copy()

                if COST_MODE == "additive_blocks":
                    x_low = model_artifacts["x_low"]
                    x_mid = model_artifacts["x_mid"]
                    x_high = model_artifacts["x_high"]

                    facility_solution["expand_low_band"] = facility_solution["facility_id"].map(lambda f: x_low[f].X)
                    facility_solution["expand_mid_band"] = facility_solution["facility_id"].map(lambda f: x_mid[f].X)
                    facility_solution["expand_high_band"] = facility_solution["facility_id"].map(lambda f: x_high[f].X)
                else:
                    x_total = model_artifacts["x_total"]
                    z_low = model_artifacts["z_low"]
                    z_mid = model_artifacts["z_mid"]
                    z_high = model_artifacts["z_high"]

                    facility_solution["expand_low_band"] = facility_solution["facility_id"].map(lambda f: x_total[f].X if z_low[f].X > 0.5 else 0.0)
                    facility_solution["expand_mid_band"] = facility_solution["facility_id"].map(lambda f: x_total[f].X if z_mid[f].X > 0.5 else 0.0)
                    facility_solution["expand_high_band"] = facility_solution["facility_id"].map(lambda f: x_total[f].X if z_high[f].X > 0.5 else 0.0)

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

                if COST_MODE == "additive_blocks":
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

                print("Expanded facilities:", facility_solution.shape)
                display(facility_solution.head())
            """
        ),
        code(
            """
            # New-build solution table
            if has_solution:
                build_rows = []
                for candidate_id, size in cand_key_tuples:
                    if y[candidate_id, size].X > 0.5:
                        row = cand_df[
                            (cand_df["candidate_id"] == candidate_id) & (cand_df["size"] == size)
                        ].iloc[0].to_dict()
                        row["build_selected"] = int(round(y[candidate_id, size].X))
                        row["new_under5_assigned"] = g[candidate_id, size].X
                        row["under5_equipment_cost_newbuild"] = 100 * g[candidate_id, size].X
                        row["total_newbuild_related_cost"] = row["fixed_build_cost"] + row["under5_equipment_cost_newbuild"]
                        build_rows.append(row)

                new_build_solution = pd.DataFrame(build_rows)

                print("Selected new builds:", new_build_solution.shape)
                display(new_build_solution.head())
            """
        ),
        code(
            """
            # Zipcode-level solution summary
            if has_solution:
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
                for column in fill_zero_cols:
                    zipcode_solution[column] = zipcode_solution[column].fillna(0)

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

                display(zipcode_solution.head())
            """
        ),
        code(
            """
            # Objective breakdown table
            if has_solution:
                objective_breakdown = pd.DataFrame({
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
                })

                display(objective_breakdown)
            """
        ),
        code(
            """
            # Run metadata and summary stats
            if has_solution:
                size_counts = new_build_solution["size"].value_counts().to_dict() if not new_build_solution.empty else {}

                run_metadata = pd.DataFrame({
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
                        COST_MODE,
                        model.Params.TimeLimit,
                    ],
                })

                summary_stats = pd.DataFrame({
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
                })

                if not new_build_solution.empty:
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
                    build_size_summary = pd.DataFrame(
                        columns=["size", "n_new_facilities", "total_capacity", "total_under5", "fixed_build_cost"]
                    )

                display(run_metadata)
                display(summary_stats)
                display(build_size_summary)
            """
        ),
        code(
            """
            # Quick management summary
            if has_solution:
                print("Objective value:", round(model.ObjVal, 2))
                print("Expansion cost:", round(objective_breakdown.loc[objective_breakdown["component"] == "expansion_cost", "value"].iloc[0], 2))
                print("New-build cost:", round(objective_breakdown.loc[objective_breakdown["component"] == "new_build_cost", "value"].iloc[0], 2))
                print("Under-5 equipment cost:", round(objective_breakdown.loc[objective_breakdown["component"] == "under5_equipment_cost", "value"].iloc[0], 2))
                print("Expanded facilities:", len(facility_solution))
                print("New facilities:", len(new_build_solution))
                print("ZIP codes meeting all requirements:", int(zipcode_solution["all_requirements_met_after"].sum()))
            """
        ),
        code(
            """
            # Export all realistic solution files
            if has_solution:
                facility_solution.to_csv(SOLUTIONS_DIR / "facility_solution_realistic.csv", index=False)
                zipcode_solution.to_csv(SOLUTIONS_DIR / "zipcode_solution_realistic.csv", index=False)
                objective_breakdown.to_csv(SOLUTIONS_DIR / "objective_breakdown_realistic.csv", index=False)
                run_metadata.to_csv(SOLUTIONS_DIR / "run_metadata_realistic.csv", index=False)
                summary_stats.to_csv(SOLUTIONS_DIR / "summary_stats_realistic.csv", index=False)
                build_size_summary.to_csv(SOLUTIONS_DIR / "build_size_summary_realistic.csv", index=False)

                if not new_build_solution.empty:
                    new_build_solution.to_csv(SOLUTIONS_DIR / "new_build_solution_realistic.csv", index=False)
                else:
                    pd.DataFrame(
                        columns=cand_df.columns.tolist()
                        + [
                            "build_selected",
                            "new_under5_assigned",
                            "under5_equipment_cost_newbuild",
                            "total_newbuild_related_cost",
                        ]
                    ).to_csv(SOLUTIONS_DIR / "new_build_solution_realistic.csv", index=False)

                print("Realistic solution files saved to:", SOLUTIONS_DIR)
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = NOTEBOOK_METADATA
    return nb


def main():
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / "notebooks"

    nb05 = build_notebook_05()
    nb06 = build_notebook_06()

    with open(notebooks_dir / "05_parameter_estimation_realistic.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb05, f)
    with open(notebooks_dir / "06_realistic_model.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb06, f)

    print("Rebuilt realistic notebooks:")
    print("-", notebooks_dir / "05_parameter_estimation_realistic.ipynb")
    print("-", notebooks_dir / "06_realistic_model.ipynb")


if __name__ == "__main__":
    main()

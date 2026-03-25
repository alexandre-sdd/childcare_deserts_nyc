from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_cell(nb, index, source, cell_type=None):
    cell = nb.cells[index]
    if cell_type is not None and cell.cell_type != cell_type:
        raise ValueError(f"Cell {index} expected type {cell_type}, found {cell.cell_type}.")
    cell.source = source.strip() + "\n"


def update_notebook_05():
    path = PROJECT_ROOT / "notebooks" / "05_parameter_estimation_realistic.ipynb"
    nb = nbformat.read(path, as_version=4)

    set_cell(
        nb,
        5,
        """
fac_clean = pd.read_csv(SHARED_DIR / "clean_childcare_facilities.csv")
facility_geo = pd.read_csv(SHARED_DIR / "facility_geo_ready.csv")
candidate_geo = pd.read_csv(SHARED_DIR / "potential_locations_geo_ready.csv")

zip_df_ideal = pd.read_csv(IDEAL_DIR / "zipcode_demand_supply_ideal.csv")
build_options = pd.read_csv(IDEAL_DIR / "build_options_ideal.csv")

zip_df = zip_df_ideal.copy()
zip_df["zipcode"] = zip_df["zipcode"].astype(str).str.zfill(5)

for col in [
    "pop_0_5",
    "child_pop_0_12",
    "req_total",
    "req_under5",
    "existing_total",
    "existing_under5",
    "total_threshold_rule",
]:
    if col in zip_df.columns:
        zip_df[col] = pd.to_numeric(zip_df[col], errors="coerce").fillna(0)

zip_df["req_total_original"] = zip_df["req_total"].astype(int)
zip_df["req_under5_original"] = zip_df["req_under5"].astype(int)

zip_df["req_total"] = np.where(
    zip_df["child_pop_0_12"] <= 0,
    0,
    zip_df["req_total_original"],
).astype(int)
zip_df["req_under5"] = np.where(
    zip_df["pop_0_5"] <= 0,
    0,
    zip_df["req_under5_original"],
).astype(int)

zip_df["zero_child_population_adjustment"] = (
    (zip_df["req_total"] != zip_df["req_total_original"])
    | (zip_df["req_under5"] != zip_df["req_under5_original"])
).astype(int)

zip_df["gap_total"] = np.maximum(0, zip_df["req_total"] - zip_df["existing_total"])
zip_df["gap_under5"] = np.maximum(0, zip_df["req_under5"] - zip_df["existing_under5"])

if "total_threshold_rule" in zip_df.columns:
    zip_df["is_desert_total_before"] = (
        (zip_df["req_total"] > 0)
        & (zip_df["existing_total"] <= zip_df["total_threshold_rule"])
    ).astype(int)

zip_df["is_under5_short_before"] = (
    zip_df["existing_under5"] < zip_df["req_under5"]
).astype(int)
zip_df["needs_intervention_before"] = (
    (zip_df["gap_total"] > 0) | (zip_df["gap_under5"] > 0)
).astype(int)

print(
    "Realistic demand rows adjusted for zero child population:",
    int(zip_df["zero_child_population_adjustment"].sum()),
)
display(zip_df.head())
""",
        cell_type="code",
    )

    set_cell(
        nb,
        9,
        """
## Realistic expansion parameter design

For the realistic scenario, expansion is capped at **20%** of current facility capacity.
The code stores both:

- regime widths that partition integer expansions across `0-10%`, `10-15%`, and `15-20%`
- regime lower/upper bounds so the optimization model can apply the assignment's
  regime-based cost formula to the full expansion amount

This avoids understating capacity by flooring each block separately and keeps the Part 2
cost logic distinct from the Part 1 approximation.
""",
        cell_type="markdown",
    )

    set_cell(
        nb,
        10,
        """
cap_10 = np.floor(0.10 * fac_existing["n_f"]).astype(int)
cap_15 = np.floor(0.15 * fac_existing["n_f"]).astype(int)
cap_20 = np.floor(0.20 * fac_existing["n_f"]).astype(int)

fac_existing["cap1"] = cap_10
fac_existing["cap2"] = np.maximum(0, cap_15 - cap_10).astype(int)
fac_existing["cap3"] = np.maximum(0, cap_20 - cap_15).astype(int)
fac_existing["U_f_realistic"] = cap_20

fac_existing["regime1_lb"] = np.where(fac_existing["cap1"] > 0, 1, 0).astype(int)
fac_existing["regime1_ub"] = fac_existing["cap1"].astype(int)
fac_existing["regime2_lb"] = np.where(fac_existing["cap2"] > 0, cap_10 + 1, 0).astype(int)
fac_existing["regime2_ub"] = (fac_existing["cap1"] + fac_existing["cap2"]).astype(int)
fac_existing["regime3_lb"] = np.where(
    fac_existing["cap3"] > 0,
    fac_existing["cap1"] + fac_existing["cap2"] + 1,
    0,
).astype(int)
fac_existing["regime3_ub"] = fac_existing["U_f_realistic"].astype(int)

fac_existing["coef1"] = (20000 + 200 * fac_existing["n_f"]) / fac_existing["n_f"]
fac_existing["coef2"] = (20000 + 400 * fac_existing["n_f"]) / fac_existing["n_f"]
fac_existing["coef3"] = (20000 + 1000 * fac_existing["n_f"]) / fac_existing["n_f"]
fac_existing["missing_geo_flag"] = fac_existing[["latitude", "longitude"]].isna().any(axis=1).astype(int)

facility_expansion_params_realistic = fac_existing[
    [
        "facility_id",
        "facility_name",
        "program_type",
        "zipcode",
        "latitude",
        "longitude",
        "missing_geo_flag",
        "n_f",
        "b_f",
        "cap1",
        "cap2",
        "cap3",
        "U_f_realistic",
        "regime1_lb",
        "regime1_ub",
        "regime2_lb",
        "regime2_ub",
        "regime3_lb",
        "regime3_ub",
        "coef1",
        "coef2",
        "coef3",
    ]
].copy()

display(facility_expansion_params_realistic.head())
""",
        cell_type="code",
    )

    set_cell(
        nb,
        21,
        """
realistic_parameter_summary = pd.DataFrame({
    "metric": [
        "n_zipcodes",
        "n_zipcodes_adjusted_for_zero_population",
        "n_existing_facilities_expandable",
        "citywide_existing_total_capacity",
        "citywide_existing_under5_capacity",
        "citywide_realistic_expansion_capacity",
        "n_existing_facilities_missing_geo",
        "n_candidate_sites",
        "n_blocked_candidate_sites",
        "n_feasible_candidate_sites",
        "n_candidate_candidate_conflicts",
        "n_candidate_existing_conflicts",
        "n_existing_existing_too_close_diagnostic",
    ],
    "value": [
        zip_df["zipcode"].nunique(),
        int(zip_df["zero_child_population_adjustment"].sum()),
        facility_expansion_params_realistic["facility_id"].nunique(),
        facility_expansion_params_realistic["n_f"].sum(),
        facility_expansion_params_realistic["b_f"].sum(),
        facility_expansion_params_realistic["U_f_realistic"].sum(),
        int(facility_expansion_params_realistic["missing_geo_flag"].sum()),
        candidate_geo_realistic["candidate_id"].nunique(),
        candidate_geo_realistic["blocked_by_existing"].sum(),
        (candidate_geo_realistic["blocked_by_existing"] == 0).sum(),
        len(candidate_candidate_conflicts),
        len(candidate_existing_conflicts),
        len(existing_existing_too_close),
    ]
})

display(realistic_parameter_summary)
""",
        cell_type="code",
    )

    set_cell(
        nb,
        22,
        """
facility_expansion_params_realistic.to_csv(
    REALISTIC_DIR / "facility_expansion_params_realistic.csv",
    index=False
)

zip_df.to_csv(
    REALISTIC_DIR / "zipcode_demand_supply_realistic.csv",
    index=False
)

candidate_geo_realistic.to_csv(
    REALISTIC_DIR / "candidate_sites_realistic.csv",
    index=False
)

candidate_build_options_feasible.to_csv(
    REALISTIC_DIR / "candidate_build_options_realistic.csv",
    index=False
)

candidate_candidate_conflicts.to_csv(
    REALISTIC_DIR / "candidate_candidate_conflicts.csv",
    index=False
)

candidate_existing_conflicts.to_csv(
    REALISTIC_DIR / "candidate_existing_conflicts.csv",
    index=False
)

existing_existing_too_close.to_csv(
    REALISTIC_DIR / "existing_existing_too_close_diagnostic.csv",
    index=False
)

realistic_parameter_summary.to_csv(
    REALISTIC_DIR / "realistic_parameter_summary.csv",
    index=False
)

print("Realistic parameter files saved to:", REALISTIC_DIR)
""",
        cell_type="code",
    )

    set_cell(
        nb,
        23,
        """
REALISTIC_ASSUMPTIONS = {
    "distance_threshold_miles": 0.06,
    "distance_enforced_within_zipcode_only": True,
    "existing_existing_conflicts_are_diagnostic_only": True,
    "candidate_existing_conflict_blocks_candidate": True,
    "use_under5_capacity_current_for_existing_capacity": True,
    "new_build_costs_and_capacities_reused_from_ideal_model": True,
    "realistic_demand_adjusts_zero_child_population_rows": True,
    "realistic_total_cap_rule": "x_f <= floor(0.20 * n_f)",
    "realistic_piecewise_cost_rule": "assignment regime-based total expansion cost",
    "expansion_piecewise_regimes": [
        "1_to_floor(0.10*n_f)",
        "floor(0.10*n_f)+1_to_floor(0.15*n_f)",
        "floor(0.15*n_f)+1_to_floor(0.20*n_f)",
    ],
    "existing_facilities_with_missing_geo_are_not_blocking_candidates": True,
}

with open(REALISTIC_DIR / "assumptions_realistic.json", "w", encoding="utf-8") as f:
    json.dump(REALISTIC_ASSUMPTIONS, f, ensure_ascii=False, indent=2)

pd.DataFrame({
    "assumption": list(REALISTIC_ASSUMPTIONS.keys()),
    "value": list(REALISTIC_ASSUMPTIONS.values())
}).to_csv(REALISTIC_DIR / "assumptions_realistic.csv", index=False)

print("Saved assumptions_realistic.json and assumptions_realistic.csv")
""",
        cell_type="code",
    )

    nbformat.write(nb, path)


def update_notebook_06():
    path = PROJECT_ROOT / "notebooks" / "06_realistic_model.ipynb"
    nb = nbformat.read(path, as_version=4)

    set_cell(
        nb,
        4,
        """
required_files = [
    REALISTIC_DIR / "zipcode_demand_supply_realistic.csv",
    REALISTIC_DIR / "facility_expansion_params_realistic.csv",
    REALISTIC_DIR / "candidate_build_options_realistic.csv",
    REALISTIC_DIR / "candidate_candidate_conflicts.csv",
    REALISTIC_DIR / "candidate_existing_conflicts.csv",
]

missing_files = [str(f) for f in required_files if not f.exists()]

print("Required input files check:")
for f in required_files:
    print(f"  {f.name}: {f.exists()}")

if missing_files:
    raise FileNotFoundError(
        "The following required files are missing. Run the realistic parameter notebook first:\\n"
        + "\\n".join(missing_files)
    )
""",
        cell_type="code",
    )

    set_cell(
        nb,
        5,
        """
zip_df = pd.read_csv(REALISTIC_DIR / "zipcode_demand_supply_realistic.csv")
fac_df = pd.read_csv(REALISTIC_DIR / "facility_expansion_params_realistic.csv")
cand_df = pd.read_csv(REALISTIC_DIR / "candidate_build_options_realistic.csv")
cc_conflict_df = pd.read_csv(REALISTIC_DIR / "candidate_candidate_conflicts.csv")
ce_conflict_df = pd.read_csv(REALISTIC_DIR / "candidate_existing_conflicts.csv")
""",
        cell_type="code",
    )

    set_cell(
        nb,
        6,
        """
zip_df["zipcode"] = zip_df["zipcode"].astype(str).str.zfill(5)
fac_df["zipcode"] = fac_df["zipcode"].astype(str).str.zfill(5)
cand_df["zipcode"] = cand_df["zipcode"].astype(str).str.zfill(5)

fac_df["facility_id"] = fac_df["facility_id"].astype(str)
cand_df["candidate_id"] = pd.to_numeric(cand_df["candidate_id"], errors="coerce").astype(int)
cand_df["size"] = cand_df["size"].astype(str)

if not cc_conflict_df.empty:
    cc_conflict_df["zipcode"] = cc_conflict_df["zipcode"].astype(str).str.zfill(5)
    cc_conflict_df["candidate_id_1"] = pd.to_numeric(cc_conflict_df["candidate_id_1"], errors="coerce").astype(int)
    cc_conflict_df["candidate_id_2"] = pd.to_numeric(cc_conflict_df["candidate_id_2"], errors="coerce").astype(int)

if not ce_conflict_df.empty:
    ce_conflict_df["zipcode"] = ce_conflict_df["zipcode"].astype(str).str.zfill(5)
    ce_conflict_df["candidate_id"] = pd.to_numeric(ce_conflict_df["candidate_id"], errors="coerce").astype(int)
    ce_conflict_df["facility_id"] = ce_conflict_df["facility_id"].astype(str)

numeric_cols_zip = ["req_total", "req_under5", "existing_total", "existing_under5", "gap_total", "gap_under5"]
for c in numeric_cols_zip:
    if c in zip_df.columns:
        zip_df[c] = pd.to_numeric(zip_df[c], errors="coerce").fillna(0).astype(int)

numeric_cols_fac = [
    "n_f",
    "b_f",
    "cap1",
    "cap2",
    "cap3",
    "U_f_realistic",
    "regime1_lb",
    "regime1_ub",
    "regime2_lb",
    "regime2_ub",
    "regime3_lb",
    "regime3_ub",
    "coef1",
    "coef2",
    "coef3",
]
for c in numeric_cols_fac:
    if c in fac_df.columns:
        fac_df[c] = pd.to_numeric(fac_df[c], errors="coerce").fillna(0)

numeric_cols_cand = ["new_total_capacity", "new_under5_capacity_max", "fixed_build_cost"]
for c in numeric_cols_cand:
    if c in cand_df.columns:
        cand_df[c] = pd.to_numeric(cand_df[c], errors="coerce").fillna(0)

print("zip_df:", zip_df.shape)
display(zip_df.head())

print("fac_df:", fac_df.shape)
display(fac_df.head())

print("cand_df:", cand_df.shape)
display(cand_df.head())

print("Candidate-candidate conflicts:", cc_conflict_df.shape)
print("Candidate-existing conflicts:", ce_conflict_df.shape)
""",
        cell_type="code",
    )

    set_cell(
        nb,
        7,
        """
required_zip_cols = ["zipcode", "req_total", "req_under5", "existing_total", "existing_under5"]
required_fac_cols = [
    "facility_id",
    "zipcode",
    "n_f",
    "b_f",
    "regime1_lb",
    "regime1_ub",
    "regime2_lb",
    "regime2_ub",
    "regime3_lb",
    "regime3_ub",
    "coef1",
    "coef2",
    "coef3",
]
required_cand_cols = ["candidate_id", "zipcode", "size", "new_total_capacity", "new_under5_capacity_max", "fixed_build_cost"]

missing_zip_cols = [c for c in required_zip_cols if c not in zip_df.columns]
missing_fac_cols = [c for c in required_fac_cols if c not in fac_df.columns]
missing_cand_cols = [c for c in required_cand_cols if c not in cand_df.columns]

if missing_zip_cols:
    raise KeyError(f"Missing columns in zipcode_demand_supply_realistic.csv: {missing_zip_cols}")
if missing_fac_cols:
    raise KeyError(f"Missing columns in facility_expansion_params_realistic.csv: {missing_fac_cols}")
if missing_cand_cols:
    raise KeyError(f"Missing columns in candidate_build_options_realistic.csv: {missing_cand_cols}")
""",
        cell_type="code",
    )

    set_cell(
        nb,
        8,
        """
Z = zip_df["zipcode"].tolist()
F = fac_df["facility_id"].tolist()
candidate_sites = sorted(cand_df["candidate_id"].unique().tolist())
S = sorted(cand_df["size"].unique().tolist())

req_total = dict(zip(zip_df["zipcode"], zip_df["req_total"]))
req_under5 = dict(zip(zip_df["zipcode"], zip_df["req_under5"]))
existing_total = dict(zip(zip_df["zipcode"], zip_df["existing_total"]))
existing_under5 = dict(zip(zip_df["zipcode"], zip_df["existing_under5"]))

fac_zip = dict(zip(fac_df["facility_id"], fac_df["zipcode"]))
n_f = dict(zip(fac_df["facility_id"], fac_df["n_f"]))
b_f = dict(zip(fac_df["facility_id"], fac_df["b_f"]))
regime1_lb = dict(zip(fac_df["facility_id"], fac_df["regime1_lb"].astype(int)))
regime1_ub = dict(zip(fac_df["facility_id"], fac_df["regime1_ub"].astype(int)))
regime2_lb = dict(zip(fac_df["facility_id"], fac_df["regime2_lb"].astype(int)))
regime2_ub = dict(zip(fac_df["facility_id"], fac_df["regime2_ub"].astype(int)))
regime3_lb = dict(zip(fac_df["facility_id"], fac_df["regime3_lb"].astype(int)))
regime3_ub = dict(zip(fac_df["facility_id"], fac_df["regime3_ub"].astype(int)))
coef1 = dict(zip(fac_df["facility_id"], fac_df["coef1"]))
coef2 = dict(zip(fac_df["facility_id"], fac_df["coef2"]))
coef3 = dict(zip(fac_df["facility_id"], fac_df["coef3"]))

cand_key_tuples = list(zip(cand_df["candidate_id"], cand_df["size"]))

build_total = {(r["candidate_id"], r["size"]): r["new_total_capacity"] for _, r in cand_df.iterrows()}
build_under5_max = {(r["candidate_id"], r["size"]): r["new_under5_capacity_max"] for _, r in cand_df.iterrows()}
build_cost = {(r["candidate_id"], r["size"]): r["fixed_build_cost"] for _, r in cand_df.iterrows()}
cand_zip = {(r["candidate_id"], r["size"]): r["zipcode"] for _, r in cand_df.iterrows()}

fac_by_zip = fac_df.groupby("zipcode")["facility_id"].apply(list).to_dict()
candsize_by_zip = cand_df.groupby("zipcode").apply(
    lambda g: list(zip(g["candidate_id"], g["size"]))
).to_dict()
sizes_by_candidate = cand_df.groupby("candidate_id")["size"].apply(list).to_dict()

blocked_candidates = set(ce_conflict_df["candidate_id"].unique().tolist()) if not ce_conflict_df.empty else set()

print("Number of zipcodes:", len(Z))
print("Number of facilities:", len(F))
print("Number of candidate sites:", len(candidate_sites))
print("Number of candidate-size options:", len(cand_key_tuples))
print("Blocked candidates from diagnostics:", len(blocked_candidates))
""",
        cell_type="code",
    )

    set_cell(
        nb,
        9,
        """
## Decision variables

- `x1[f]`, `x2[f]`, `x3[f]`: total expansion chosen for facility `f` if it falls in the
  `0-10%`, `10-15%`, or `15-20%` cost regime
- `z1[f]`, `z2[f]`, `z3[f]`: binaries selecting at most one realistic expansion cost regime
- `u[f]`: under-5 slots assigned within expansion at facility `f`
- `y[l,s]`: binary variable equal to 1 if candidate site `l` is used for size `s`
- `g[l,s]`: under-5 slots assigned to the new facility built at candidate site `l` with size `s`
""",
        cell_type="markdown",
    )

    set_cell(
        nb,
        10,
        """
log_file = LOG_DIR / "realistic_model.log"

model = gp.Model("realistic_childcare")
model.Params.LogFile = str(log_file)
model.Params.MIPGap = 0.001
model.Params.TimeLimit = 600
model.Params.OutputFlag = 1

x1 = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x1")
x2 = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x2")
x3 = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="x3")
z1 = model.addVars(F, vtype=GRB.BINARY, name="z1")
z2 = model.addVars(F, vtype=GRB.BINARY, name="z2")
z3 = model.addVars(F, vtype=GRB.BINARY, name="z3")
u = model.addVars(F, vtype=GRB.INTEGER, lb=0, name="u")

y = model.addVars(cand_key_tuples, vtype=GRB.BINARY, name="y")
g = model.addVars(cand_key_tuples, vtype=GRB.INTEGER, lb=0, name="g")
""",
        cell_type="code",
    )

    set_cell(
        nb,
        11,
        """
for f in F:
    model.addConstr(z1[f] + z2[f] + z3[f] <= 1, name=f"one_regime_{f}")

    model.addConstr(x1[f] <= regime1_ub[f] * z1[f], name=f"regime1_ub_{f}")
    model.addConstr(x2[f] <= regime2_ub[f] * z2[f], name=f"regime2_ub_{f}")
    model.addConstr(x3[f] <= regime3_ub[f] * z3[f], name=f"regime3_ub_{f}")

    if regime1_ub[f] > 0:
        model.addConstr(x1[f] >= regime1_lb[f] * z1[f], name=f"regime1_lb_{f}")
    else:
        model.addConstr(z1[f] == 0, name=f"regime1_disabled_{f}")

    if regime2_ub[f] >= regime2_lb[f] and regime2_lb[f] > 0:
        model.addConstr(x2[f] >= regime2_lb[f] * z2[f], name=f"regime2_lb_{f}")
    else:
        model.addConstr(z2[f] == 0, name=f"regime2_disabled_{f}")

    if regime3_ub[f] >= regime3_lb[f] and regime3_lb[f] > 0:
        model.addConstr(x3[f] >= regime3_lb[f] * z3[f], name=f"regime3_lb_{f}")
    else:
        model.addConstr(z3[f] == 0, name=f"regime3_disabled_{f}")

    model.addConstr(u[f] <= x1[f] + x2[f] + x3[f], name=f"under5_expand_cap_{f}")
""",
        cell_type="code",
    )

    set_cell(
        nb,
        21,
        """
has_solution = model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0

if has_solution:
    facility_solution = fac_df.copy()
    facility_solution["x1"] = facility_solution["facility_id"].map(lambda f: x1[f].X)
    facility_solution["x2"] = facility_solution["facility_id"].map(lambda f: x2[f].X)
    facility_solution["x3"] = facility_solution["facility_id"].map(lambda f: x3[f].X)
    facility_solution["z1"] = facility_solution["facility_id"].map(lambda f: z1[f].X)
    facility_solution["z2"] = facility_solution["facility_id"].map(lambda f: z2[f].X)
    facility_solution["z3"] = facility_solution["facility_id"].map(lambda f: z3[f].X)
    facility_solution["u_under5"] = facility_solution["facility_id"].map(lambda f: u[f].X)
    facility_solution["expand_total"] = facility_solution["x1"] + facility_solution["x2"] + facility_solution["x3"]
    facility_solution["selected_regime"] = np.select(
        [
            facility_solution["z1"] > 0.5,
            facility_solution["z2"] > 0.5,
            facility_solution["z3"] > 0.5,
        ],
        [
            "0_10_pct",
            "10_15_pct",
            "15_20_pct",
        ],
        default="none",
    )
    facility_solution["expansion_cost"] = (
        facility_solution["coef1"] * facility_solution["x1"]
        + facility_solution["coef2"] * facility_solution["x2"]
        + facility_solution["coef3"] * facility_solution["x3"]
    )
    facility_solution["under5_equipment_cost_expansion"] = 100 * facility_solution["u_under5"]
    facility_solution["total_expansion_related_cost"] = (
        facility_solution["expansion_cost"] + facility_solution["under5_equipment_cost_expansion"]
    )
    facility_solution["expansion_share_of_existing"] = np.where(
        facility_solution["n_f"] > 0,
        facility_solution["expand_total"] / facility_solution["n_f"],
        0,
    )
    facility_solution = facility_solution[facility_solution["expand_total"] > 1e-6].copy()

    print("Expanded facilities:", facility_solution.shape)
    display(facility_solution.head())
""",
        cell_type="code",
    )

    nbformat.write(nb, path)


def main():
    update_notebook_05()
    update_notebook_06()
    print("Updated Part 2 notebooks.")


if __name__ == "__main__":
    main()

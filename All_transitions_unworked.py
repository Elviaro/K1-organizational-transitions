from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("dataset_audit_result.xlsx")

STRUCTURE_COLUMNS = [
    "vp",
    "svp",
    "evp",
    "sevp",
    "dir",
    "sdir",
    "md",
    "smd",
    "se",
    "vc",
    "svc",
    "president",
    "board",
    "ceo",
    "cxo",
    "primary",
    "support",
    "bu",
]

EXPECTED_COLUMNS = [
    "year",
    "company",
    "GV_KEY",
    "ticker",
    "cusip",
    "GICGroups",
    "GICIndustries",
    "GICSectors",
    "GICSubIndustries",
    "sic_code",
    "CIK",
    "role",
    "last_name",
    "first_name",
    "full_name",
    "uniqueid",
    "TMTSource",
    "marg_note1",
    *STRUCTURE_COLUMNS,
]


# ============================================================
# LOADING
# ============================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """Load the source Excel dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    print(f"Loading dataset: {path}")

    df = pd.read_excel(path)

    print(
        f"Loaded: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Check whether expected columns exist."""

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    print("Column validation: OK")

    if extra:
        print("Additional columns:")
        for column in extra:
            print(f"  - {column}")


# ============================================================
# MISSING VALUES
# ============================================================

def analyze_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate missing values for every column."""

    result = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })

    result["missing_percent"] = (
        result["missing_count"]
        / len(df)
        * 100
    )

    return result.sort_values(
        "missing_count",
        ascending=False,
    )


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_basic_statistics(
    df: pd.DataFrame,
) -> dict:
    """Calculate basic dataset statistics."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "companies": df["company"].nunique(),
        "gv_keys": df["GV_KEY"].nunique(),
        "years": df["year"].nunique(),
        "min_year": df["year"].min(),
        "max_year": df["year"].max(),
        "unique_people": df["uniqueid"].nunique(),
    }


def print_basic_statistics(stats: dict) -> None:
    """Print basic dataset statistics."""

    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    print(f"Rows:             {stats['rows']:,}")
    print(f"Columns:          {stats['columns']:,}")
    print(f"Companies:        {stats['companies']:,}")
    print(f"GV_KEY:           {stats['gv_keys']:,}")
    print(f"Unique years:     {stats['years']:,}")
    print(f"Year range:       {stats['min_year']}–{stats['max_year']}")
    print(f"Unique uniqueid:  {stats['unique_people']:,}")


# ============================================================
# ROLE ANALYSIS
# ============================================================

def prepare_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized lowercase role values for analysis."""

    result = df.copy()

    result["role_clean"] = (
        result["role"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return result


def analyze_roles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate role frequency statistics."""

    role_counts = (
        df["role_clean"]
        .value_counts(dropna=False)
        .reset_index()
    )

    role_counts.columns = [
        "role",
        "observations",
    ]

    role_counts["percent"] = (
        role_counts["observations"]
        / len(df)
        * 100
    )

    return role_counts


def print_role_statistics(
    df: pd.DataFrame,
    role_counts: pd.DataFrame,
) -> None:
    """Print role statistics."""

    unique_roles = df["role_clean"].nunique()

    single_roles = (
        role_counts["observations"] == 1
    ).sum()

    print("\n" + "=" * 70)
    print("ROLE ANALYSIS")
    print("=" * 70)

    print(f"Unique roles:          {unique_roles:,}")
    print(f"Single-occurrence:     {single_roles:,}")
    print(
        "Single-occurrence %:   "
        f"{single_roles / len(role_counts) * 100:.2f}%"
    )

    print("\nTop 30 roles:")
    print(
        role_counts
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# COMPANY × YEAR STATES
# ============================================================

def build_company_year_states(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build an organizational state for each company-year."""

    group = df.groupby(
        ["company", "year"],
        dropna=False,
    )

    states = group.agg(
        n_TMT=("uniqueid", "count"),
        n_unique_roles=("role_clean", "nunique"),
    ).reset_index()

    available_columns = [
        column
        for column in STRUCTURE_COLUMNS
        if column in df.columns
    ]

    structure = (
        group[available_columns]
        .sum()
        .reset_index()
    )

    states = states.merge(
        structure,
        on=["company", "year"],
        how="left",
    )

    return states.sort_values(
        ["company", "year"]
    ).reset_index(drop=True)


# ============================================================
# TRANSITIONS
# ============================================================

def build_transitions(
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Build transitions between consecutive years."""

    states = states.sort_values(
        ["company", "year"]
    ).reset_index(drop=True)

    states["next_year"] = (
        states
        .groupby("company")["year"]
        .shift(-1)
    )

    states["year_gap"] = (
        states["next_year"]
        - states["year"]
    )

    transitions = states[
        states["year_gap"] == 1
    ].copy()

    next_states = states[
        [
            "company",
            "year",
            "n_TMT",
            "n_unique_roles",
            *STRUCTURE_COLUMNS,
        ]
    ].copy()

    next_states = next_states.rename(
        columns={
            "year": "next_year",
            "n_TMT": "next_n_TMT",
            "n_unique_roles": "next_n_unique_roles",
            **{
                column: f"next_{column}"
                for column in STRUCTURE_COLUMNS
            },
        }
    )

    transitions = transitions.merge(
        next_states,
        on=["company", "next_year"],
        how="left",
    )

    return transitions


# ============================================================
# STRUCTURAL CHANGES
# ============================================================

def calculate_structure_changes(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate changes in structural indicators."""

    result = transitions.copy()

    result["delta_TMT"] = (
        result["next_n_TMT"]
        - result["n_TMT"]
    )

    result["delta_unique_roles"] = (
        result["next_n_unique_roles"]
        - result["n_unique_roles"]
    )

    for column in STRUCTURE_COLUMNS:

        if column not in result.columns:
            continue

        next_column = f"next_{column}"

        if next_column not in result.columns:
            continue

        result[f"delta_{column}"] = (
            result[next_column]
            - result[column]
        )

    return result


# ============================================================
# ROLE CHANGES
# ============================================================

def build_role_sets(
    df: pd.DataFrame,
) -> dict:
    """Build role sets for every company-year."""

    return (
        df.groupby(
            ["company", "year"]
        )["role_clean"]
        .apply(
            lambda values: {
                role
                for role in values.dropna()
                if str(role).strip()
            }
        )
        .to_dict()
    )


def calculate_role_changes(
    transitions: pd.DataFrame,
    role_sets: dict,
) -> pd.DataFrame:
    """Find roles added and removed between consecutive years."""

    result = transitions.copy()

    added_roles = []
    removed_roles = []

    for _, row in result.iterrows():

        key_current = (
            row["company"],
            row["year"],
        )

        key_next = (
            row["company"],
            row["next_year"],
        )

        current_roles = role_sets.get(
            key_current,
            set(),
        )

        next_roles = role_sets.get(
            key_next,
            set(),
        )

        added = sorted(
            next_roles - current_roles
        )

        removed = sorted(
            current_roles - next_roles
        )

        added_roles.append(
            "; ".join(added)
        )

        removed_roles.append(
            "; ".join(removed)
        )

    result["roles_added"] = added_roles
    result["roles_removed"] = removed_roles

    result["n_roles_added"] = (
        result["roles_added"]
        .str.split("; ")
        .apply(
            lambda x:
            0 if x == [""] else len(x)
        )
    )

    result["n_roles_removed"] = (
        result["roles_removed"]
        .str.split("; ")
        .apply(
            lambda x:
            0 if x == [""] else len(x)
        )
    )

    return result


# ============================================================
# BINARY FIELD ANALYSIS
# ============================================================

def summarize_binary_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize binary structural indicators."""

    summary = []

    for column in STRUCTURE_COLUMNS:

        if column not in df.columns:
            continue

        summary.append({
            "field": column,
            "n_1": int(
                df[column].eq(1).sum()
            ),
            "n_0": int(
                df[column].eq(0).sum()
            ),
            "missing": int(
                df[column].isna().sum()
            ),
            "share_1_percent":
                df[column].eq(1).mean() * 100,
        })

    return pd.DataFrame(summary)


# ============================================================
# DELTA ANALYSIS
# ============================================================

def summarize_deltas(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize increases and decreases in structural indicators."""

    summary = []

    for column in STRUCTURE_COLUMNS:

        delta_column = f"delta_{column}"

        if delta_column not in transitions.columns:
            continue

        values = transitions[delta_column]

        summary.append({
            "field": column,
            "n_decrease": int(
                (values < 0).sum()
            ),
            "n_no_change": int(
                (values == 0).sum()
            ),
            "n_increase": int(
                (values > 0).sum()
            ),
            "mean_delta": values.mean(),
            "max_increase": values.max(),
            "max_decrease": values.min(),
        })

    return pd.DataFrame(summary)


# ============================================================
# LARGE CHANGES
# ============================================================

def find_largest_changes(
    transitions: pd.DataFrame,
    limit: int = 100,
) -> pd.DataFrame:
    """Find transitions with the largest role changes."""

    result = transitions.copy()

    result["change_size"] = (
        result["n_roles_added"]
        + result["n_roles_removed"]
    )

    columns = [
        "company",
        "year",
        "next_year",
        "n_TMT",
        "next_n_TMT",
        "delta_TMT",
        "n_unique_roles",
        "next_n_unique_roles",
        "delta_unique_roles",
        "n_roles_added",
        "n_roles_removed",
        "roles_added",
        "roles_removed",
        "change_size",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    return (
        result
        .sort_values(
            "change_size",
            ascending=False,
        )
        [columns]
        .head(limit)
    )


# ============================================================
# COMPANY CHANGE SUMMARY
# ============================================================

def summarize_company_changes(
    transitions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate organizational changes by company."""

    return (
        transitions
        .groupby("company")
        .agg(
            transitions=("year", "count"),
            total_roles_added=(
                "n_roles_added",
                "sum",
            ),
            total_roles_removed=(
                "n_roles_removed",
                "sum",
            ),
            mean_delta_TMT=(
                "delta_TMT",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "total_roles_added",
            ascending=False,
        )
    )


# ============================================================
# EXPORT
# ============================================================

def export_results(
    output_path: Path,
    missing: pd.DataFrame,
    role_counts: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    largest_changes: pd.DataFrame,
    company_changes: pd.DataFrame,
    binary_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> None:
    """Export analysis results to Excel."""

    print("\nSaving results...")

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        missing.to_excel(
            writer,
            sheet_name="missing",
            index=False,
        )

        role_counts.to_excel(
            writer,
            sheet_name="role_frequency",
            index=False,
        )

        states.to_excel(
            writer,
            sheet_name="company_year",
            index=False,
        )

        transitions.to_excel(
            writer,
            sheet_name="transitions",
            index=False,
        )

        largest_changes.to_excel(
            writer,
            sheet_name="largest_changes",
            index=False,
        )

        company_changes.to_excel(
            writer,
            sheet_name="company_changes",
            index=False,
        )

        binary_summary.to_excel(
            writer,
            sheet_name="binary_summary",
            index=False,
        )

        delta_summary.to_excel(
            writer,
            sheet_name="delta_summary",
            index=False,
        )

    print(
        f"Results saved to: "
        f"{output_path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("TMT DATASET AUDIT")
    print("=" * 70)

    # 1. Load
    df = load_dataset(INPUT_FILE)

    # 2. Validate
    validate_columns(df)

    # 3. Prepare roles
    df = prepare_roles(df)

    # 4. Basic statistics
    stats = calculate_basic_statistics(df)
    print_basic_statistics(stats)

    # 5. Missing values
    print("\nAnalyzing missing values...")
    missing = analyze_missing_values(df)

    # 6. Roles
    print("\nAnalyzing roles...")
    role_counts = analyze_roles(df)
    print_role_statistics(
        df,
        role_counts,
    )

    # 7. Company × year states
    print("\nBuilding company-year states...")
    states = build_company_year_states(df)

    print(
        f"Company-year states: "
        f"{len(states):,}"
    )

    # 8. Transitions
    print("\nBuilding yearly transitions...")
    transitions = build_transitions(states)

    print(
        f"Yearly transitions: "
        f"{len(transitions):,}"
    )

    # 9. Structural changes
    print("\nCalculating structural changes...")
    transitions = calculate_structure_changes(
        transitions
    )

    # 10. Role changes
    print("\nCalculating role changes...")
    role_sets = build_role_sets(df)

    transitions = calculate_role_changes(
        transitions,
        role_sets,
    )

    # 11. Binary summary
    print("\nAnalyzing binary indicators...")
    binary_summary = summarize_binary_fields(df)

    # 12. Delta summary
    print("\nAnalyzing structural deltas...")
    delta_summary = summarize_deltas(
        transitions
    )

    # 13. Largest changes
    largest_changes = find_largest_changes(
        transitions
    )

    # 14. Company summary
    company_changes = summarize_company_changes(
        transitions
    )

    # 15. Export
    export_results(
        OUTPUT_FILE,
        missing,
        role_counts,
        states,
        transitions,
        largest_changes,
        company_changes,
        binary_summary,
        delta_summary,
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("person_profiles.xlsx")

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

PERSON_COLUMNS = [
    "company",
    "GV_KEY",
    "ticker",
    "CIK",
    "last_name",
    "first_name",
    "full_name",
    "uniqueid",
    "TMTSource",
    "marg_note1",
]

REQUIRED_COLUMNS = [
    "year",
    "company",
    "GV_KEY",
    "ticker",
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
    """Check required columns."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print("Column validation: OK")


# ============================================================
# PREPARATION
# ============================================================

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare fields required for person profiling.
    """

    result = df.copy()

    # Clean names.
    for column in [
        "first_name",
        "last_name",
        "full_name",
    ]:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

    # Clean role.
    result["role_clean"] = (
        result["role"]
        .astype("string")
        .str.strip()
    )

    # Normalize year.
    result["year"] = pd.to_numeric(
        result["year"],
        errors="coerce",
    )

    return result


# ============================================================
# PERSON IDENTIFICATION
# ============================================================

def build_person_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a preliminary person key.

    Current rule:
        company + full_name

    We keep uniqueid separately because it identifies
    the original observation and must not be lost.
    """

    result = df.copy()

    result["person_key"] = (
        result["company"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        + "||"
        + result["full_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return result


# ============================================================
# PERSON DIAGNOSTICS
# ============================================================

def analyze_person_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check how many source records are grouped into
    each preliminary person profile.
    """

    result = (
        df.groupby("person_key")
        .agg(
            records=("uniqueid", "count"),
            companies=("company", "nunique"),
            names=("full_name", "nunique"),
            years=("year", "nunique"),
        )
        .reset_index()
    )

    return result


def find_suspicious_person_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find profiles that may contain identification problems.
    """

    grouped = (
        df.groupby("person_key")
        .agg(
            records=("uniqueid", "count"),
            companies=("company", "nunique"),
            names=("full_name", "nunique"),
        )
        .reset_index()
    )

    suspicious = grouped[
        (grouped["companies"] > 1)
        | (grouped["names"] > 1)
    ].copy()

    return suspicious.sort_values(
        "records",
        ascending=False,
    )


# ============================================================
# ROLE HISTORY
# ============================================================

def build_role_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create detailed chronological role history.

    One row = one original dataset observation.
    """

    columns = [
        "person_key",
        "year",
        *PERSON_COLUMNS,
        "role",
        "role_clean",
        *STRUCTURE_COLUMNS,
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    history = df[columns].copy()

    history = history.sort_values(
        [
            "person_key",
            "year",
            "role_clean",
        ],
        na_position="last",
    )

    history["role_number"] = (
        history
        .groupby("person_key")
        .cumcount()
        + 1
    )

    return history.reset_index(drop=True)


# ============================================================
# ROLE TRANSITIONS
# ============================================================

def build_role_transitions(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build transitions between consecutive observations
    within each person profile.
    """

    result = history.copy()

    result["previous_year"] = (
        result
        .groupby("person_key")["year"]
        .shift(1)
    )

    result["previous_role"] = (
        result
        .groupby("person_key")["role_clean"]
        .shift(1)
    )

    result["year_gap"] = (
        result["year"]
        - result["previous_year"]
    )

    result["role_changed"] = (
        result["role_clean"]
        != result["previous_role"]
    )

    return result


# ============================================================
# PROFILE BUILDING
# ============================================================

def build_person_profiles(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one compact row per person/company profile.

    The complete role trajectory is stored as a text sequence.
    """

    profiles = (
        history
        .groupby("person_key")
        .agg(
            company=("company", "first"),
            GV_KEY=("GV_KEY", "first"),
            ticker=("ticker", "first"),
            CIK=("CIK", "first"),
            full_name=("full_name", "first"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_records=("uniqueid", "count"),
            n_years=("year", "nunique"),
            n_roles=("role_clean", "nunique"),
        )
        .reset_index()
    )

    role_sequences = (
        history
        .sort_values(
            ["person_key", "year"]
        )
        .groupby("person_key")
        .agg(
            role_history=(
                "role_clean",
                lambda values:
                " → ".join(
                    str(value)
                    for value in values
                    if pd.notna(value)
                    and str(value).strip()
                )
            )
        )
        .reset_index()
    )

    profiles = profiles.merge(
        role_sequences,
        on="person_key",
        how="left",
    )

    return profiles.sort_values(
        ["company", "full_name"]
    ).reset_index(drop=True)


# ============================================================
# STRUCTURAL TRAJECTORY
# ============================================================

def build_structure_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the structural 18-field vector for every
    person observation.
    """

    columns = [
        "person_key",
        "year",
        "role",
        *STRUCTURE_COLUMNS,
    ]

    columns = [
        column
        for column in columns
        if column in history.columns
    ]

    return history[columns].copy()


# ============================================================
# EXPORT
# ============================================================

def export_results(
    output_path: Path,
    profiles: pd.DataFrame,
    history: pd.DataFrame,
    transitions: pd.DataFrame,
    suspicious: pd.DataFrame,
    structure_history: pd.DataFrame,
) -> None:
    """Save all profile-related results."""

    print("\nSaving results...")

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        profiles.to_excel(
            writer,
            sheet_name="person_profiles",
            index=False,
        )

        history.to_excel(
            writer,
            sheet_name="role_history",
            index=False,
        )

        transitions.to_excel(
            writer,
            sheet_name="role_transitions",
            index=False,
        )

        structure_history.to_excel(
            writer,
            sheet_name="structure_history",
            index=False,
        )

        suspicious.to_excel(
            writer,
            sheet_name="suspicious_profiles",
            index=False,
        )

    print(
        f"Results saved to:\n"
        f"{output_path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("PERSON PROFILE BUILDER")
    print("=" * 70)

    # 1. Load
    df = load_dataset(INPUT_FILE)

    # 2. Validate
    validate_columns(df)

    # 3. Prepare
    df = prepare_dataset(df)

    # 4. Build preliminary person key
    print("\nBuilding person keys...")
    df = build_person_key(df)

    # 5. Diagnostics
    print("\nAnalyzing person keys...")
    person_diagnostics = analyze_person_keys(df)

    suspicious = find_suspicious_person_keys(df)

    print(
        f"Preliminary profiles: "
        f"{len(person_diagnostics):,}"
    )

    print(
        f"Suspicious profiles: "
        f"{len(suspicious):,}"
    )

    # 6. Detailed role history
    print("\nBuilding role history...")
    history = build_role_history(df)

    # 7. Role transitions
    print("\nBuilding role transitions...")
    transitions = build_role_transitions(history)

    # 8. Compact profiles
    print("\nBuilding person profiles...")
    profiles = build_person_profiles(history)

    # 9. Structural history
    print("\nBuilding structural history...")
    structure_history = build_structure_history(history)

    # 10. Export
    export_results(
        OUTPUT_FILE,
        profiles,
        history,
        transitions,
        suspicious,
        structure_history,
    )

    print("\n" + "=" * 70)
    print("PROFILE BUILDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path


# ============================================================
# НАСТРОЙКИ
# ============================================================

INPUT_FILE = Path("20260721_OD_Database_1993_2024b.xlsx")
OUTPUT_FILE = Path("dataset_audit.xlsx")

# Сколько наиболее частых должностей показывать
TOP_N_ROLES = 100

# Бинарные признаки структуры TMT
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


# ============================================================
# 1. ЗАГРУЗКА
# ============================================================

print("=" * 70)
print("ЗАГРУЗКА ДАТАСЕТА")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Файл не найден: {INPUT_FILE.resolve()}"
    )

df = pd.read_excel(INPUT_FILE)

print(f"\nФайл: {INPUT_FILE}")
print(f"Размер таблицы: {df.shape[0]:,} строк × {df.shape[1]} столбцов")

print("\nСтолбцы:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")


# ============================================================
# 2. ПРОВЕРКА ОЖИДАЕМЫХ ПОЛЕЙ
# ============================================================

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
] + STRUCTURE_COLUMNS

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

extra_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

print("\nПроверка структуры:")

if missing_columns:
    print("ОТСУТСТВУЮТ:")
    for col in missing_columns:
        print(f"  - {col}")
else:
    print("Все ожидаемые поля присутствуют.")

if extra_columns:
    print("\nДополнительные поля:")
    for col in extra_columns:
        print(f"  - {col}")


# ============================================================
# 3. ПРОПУСКИ
# ============================================================

print("\n" + "=" * 70)
print("ПРОПУСКИ")
print("=" * 70)

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
})

missing["missing_percent"] = (
    missing["missing_count"] / len(df) * 100
)

missing = missing.sort_values(
    "missing_count",
    ascending=False
)

print(missing.to_string(index=False))


# ============================================================
# 4. БАЗОВАЯ СТАТИСТИКА
# ============================================================

print("\n" + "=" * 70)
print("БАЗОВАЯ СТАТИСТИКА")
print("=" * 70)

print(f"Количество строк: {len(df):,}")

if "company" in df.columns:
    print(
        f"Уникальных компаний: "
        f"{df['company'].nunique():,}"
    )

if "GV_KEY" in df.columns:
    print(
        f"Уникальных GV_KEY: "
        f"{df['GV_KEY'].nunique():,}"
    )

if "year" in df.columns:
    print(
        f"Минимальный год: "
        f"{df['year'].min()}"
    )
    print(
        f"Максимальный год: "
        f"{df['year'].max()}"
    )
    print(
        f"Количество лет: "
        f"{df['year'].nunique()}"
    )

if "uniqueid" in df.columns:
    print(
        f"Уникальных руководителей: "
        f"{df['uniqueid'].nunique():,}"
    )


# ============================================================
# 5. УНИКАЛЬНЫЕ ROLE
# ============================================================

print("\n" + "=" * 70)
print("АНАЛИЗ ROLE")
print("=" * 70)

df["role_clean"] = (
    df["role"]
    .astype("string")
    .str.strip()
    .str.lower()
)

role_counts = (
    df["role_clean"]
    .value_counts(dropna=False)
    .reset_index()
)

role_counts.columns = [
    "role",
    "observations"
]

role_counts["percent"] = (
    role_counts["observations"]
    / len(df)
    * 100
)

print(
    f"\nУникальных значений role: "
    f"{df['role_clean'].nunique():,}"
)

single_roles = (
    role_counts["observations"] == 1
).sum()

print(
    f"Ролей, встречающихся только один раз: "
    f"{single_roles:,}"
)

print(
    f"Доля уникальных одноразовых ролей: "
    f"{single_roles / len(role_counts) * 100:.2f}%"
)

print("\nТОП ролей:")
print(
    role_counts.head(TOP_N_ROLES)
    .to_string(index=False)
)


# ============================================================
# 6. КОЛИЧЕСТВО ЧЛЕНОВ TMT ПО КОМПАНИИ И ГОДУ
# ============================================================

print("\n" + "=" * 70)
print("ОРГАНИЗАЦИОННЫЕ СОСТОЯНИЯ COMPANY × YEAR")
print("=" * 70)

group = (
    df.groupby(["company", "year"], dropna=False)
)

states = group.agg(
    n_TMT=("uniqueid", "nunique"),
    n_rows=("uniqueid", "size"),
    n_unique_roles=("role_clean", "nunique"),
).reset_index()

print(f"\nКоличество состояний company × year: {len(states):,}")

print("\nПервые состояния:")
print(states.head(20).to_string(index=False))


# ============================================================
# 7. АГРЕГАЦИЯ СТРУКТУРНЫХ ФЛАГОВ
# ============================================================

print("\n" + "=" * 70)
print("АГРЕГАЦИЯ СТРУКТУРНЫХ ПРИЗНАКОВ")
print("=" * 70)

available_structure_columns = [
    col for col in STRUCTURE_COLUMNS
    if col in df.columns
]

structure_agg = group[available_structure_columns].sum()

structure_agg = structure_agg.reset_index()

states = states.merge(
    structure_agg,
    on=["company", "year"],
    how="left"
)

print("\nПример:")
print(
    states.head(20).to_string(index=False)
)


# ============================================================
# 8. ИЗМЕНЕНИЯ МЕЖДУ СОСЕДНИМИ ГОДАМИ
# ============================================================

print("\n" + "=" * 70)
print("ИЗМЕНЕНИЯ COMPANY: YEAR → NEXT YEAR")
print("=" * 70)

states = states.sort_values(
    ["company", "year"]
).reset_index(drop=True)

states["next_year"] = (
    states.groupby("company")["year"]
    .shift(-1)
)

states["year_gap"] = (
    states["next_year"] - states["year"]
)

# Берем только действительно соседние годы
transitions = states[
    states["year_gap"] == 1
].copy()

print(
    f"\nПереходов между соседними годами: "
    f"{len(transitions):,}"
)


# ============================================================
# 9. Δ СТРУКТУРНЫХ ПРИЗНАКОВ
# ============================================================

for col in available_structure_columns:
    transitions[f"delta_{col}"] = (
        transitions.groupby("company")[col]
        .shift(-1)
        - transitions[col]
    )

# Исправляем логику: shift должен быть рассчитан по исходным состояниям
# Поэтому создаем таблицу следующего года отдельно.

next_states = states[
    ["company", "year"] +
    ["n_TMT", "n_unique_roles"] +
    available_structure_columns
].copy()

next_states = next_states.rename(
    columns={
        "year": "next_year",
        "n_TMT": "next_n_TMT",
        "n_unique_roles": "next_n_unique_roles",
        **{
            col: f"next_{col}"
            for col in available_structure_columns
        }
    }
)

transitions = transitions.drop(
    columns=[
        col for col in transitions.columns
        if col.startswith("delta_")
    ],
    errors="ignore"
)

transitions = transitions.merge(
    next_states,
    on=["company", "next_year"],
    how="left"
)

transitions["delta_TMT"] = (
    transitions["next_n_TMT"]
    - transitions["n_TMT"]
)

transitions["delta_unique_roles"] = (
    transitions["next_n_unique_roles"]
    - transitions["n_unique_roles"]
)

for col in available_structure_columns:
    transitions[f"delta_{col}"] = (
        transitions[f"next_{col}"]
        - transitions[col]
    )


# ============================================================
# 10. МНОЖЕСТВА ПОЯВИВШИХСЯ И ИСЧЕЗНУВШИХ ROLE
# ============================================================

print("\nСчитаем появившиеся и исчезнувшие роли...")

# Словарь:
# (company, year) -> множество ролей

role_sets = (
    df.groupby(["company", "year"])["role_clean"]
    .apply(
        lambda x: set(
            r for r in x.dropna()
            if str(r).strip()
        )
    )
    .to_dict()
)

added_roles = []
removed_roles = []

for _, row in transitions.iterrows():

    company = row["company"]
    year = row["year"]
    next_year = row["next_year"]

    old_roles = role_sets.get(
        (company, year),
        set()
    )

    new_roles = role_sets.get(
        (company, next_year),
        set()
    )

    added = sorted(new_roles - old_roles)
    removed = sorted(old_roles - new_roles)

    added_roles.append("; ".join(added))
    removed_roles.append("; ".join(removed))

transitions["roles_added"] = added_roles
transitions["roles_removed"] = removed_roles

transitions["n_roles_added"] = (
    transitions["roles_added"]
    .apply(
        lambda x: 0 if not x else len(x.split("; "))
    )
)

transitions["n_roles_removed"] = (
    transitions["roles_removed"]
    .apply(
        lambda x: 0 if not x else len(x.split("; "))
    )
)

# Потенциальный индикатор структурного изменения.
# Пока это ТОЛЬКО диагностический показатель.
transitions["structural_change_raw"] = (
    (
        transitions["n_roles_added"] > 0
    )
    |
    (
        transitions["n_roles_removed"] > 0
    )
    |
    (
        transitions["delta_TMT"] != 0
    )
)


# ============================================================
# 11. САМЫЕ СИЛЬНЫЕ ИЗМЕНЕНИЯ
# ============================================================

print("\n" + "=" * 70)
print("НАИБОЛЬШИЕ ИЗМЕНЕНИЯ")
print("=" * 70)

interesting_columns = [
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
]

interesting_columns = [
    col for col in interesting_columns
    if col in transitions.columns
]

largest_changes = (
    transitions
    .assign(
        change_size=lambda x:
        x["n_roles_added"]
        + x["n_roles_removed"]
    )
    .sort_values(
        "change_size",
        ascending=False
    )
)

print(
    largest_changes[
        interesting_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 12. КОМПАНИИ С НАИБОЛЬШИМ ЧИСЛОМ ИЗМЕНЕНИЙ
# ============================================================

company_changes = (
    transitions
    .groupby("company")
    .agg(
        transitions=("year", "count"),
        total_roles_added=("n_roles_added", "sum"),
        total_roles_removed=("n_roles_removed", "sum"),
        mean_delta_TMT=("delta_TMT", "mean"),
    )
    .reset_index()
    .sort_values(
        "total_roles_added",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("КОМПАНИИ С НАИБОЛЬШИМ ЧИСЛОМ ПОЯВЛЕНИЙ ROLE")
print("=" * 70)

print(
    company_changes.head(30)
    .to_string(index=False)
)


# ============================================================
# 13. АНАЛИЗ BINARIES
# ============================================================

print("\n" + "=" * 70)
print("РАСПРЕДЕЛЕНИЕ БИНАРНЫХ ПРИЗНАКОВ")
print("=" * 70)

binary_summary = []

for col in available_structure_columns:

    value_counts = df[col].value_counts(
        dropna=False
    )

    binary_summary.append({
        "field": col,
        "n_1": int(value_counts.get(1, 0)),
        "n_0": int(value_counts.get(0, 0)),
        "missing": int(df[col].isna().sum()),
        "share_1_percent":
            df[col].eq(1).mean() * 100
    })

binary_summary = pd.DataFrame(
    binary_summary
)

print(
    binary_summary
    .to_string(index=False)
)


# ============================================================
# 14. ПОТЕНЦИАЛЬНЫЕ ИЗМЕНЕНИЯ В ИЕРАРХИИ
# ============================================================

print("\n" + "=" * 70)
print("ЧАСТОТА ИЗМЕНЕНИЙ СТРУКТУРНЫХ КАТЕГОРИЙ")
print("=" * 70)

delta_columns = [
    f"delta_{col}"
    for col in available_structure_columns
]

delta_summary = []

for col in delta_columns:

    delta_summary.append({
        "field": col.replace("delta_", ""),
        "n_decrease":
            int((transitions[col] < 0).sum()),
        "n_no_change":
            int((transitions[col] == 0).sum()),
        "n_increase":
            int((transitions[col] > 0).sum()),
        "mean_delta":
            transitions[col].mean(),
        "max_increase":
            transitions[col].max(),
        "max_decrease":
            transitions[col].min(),
    })

delta_summary = pd.DataFrame(
    delta_summary
)

print(
    delta_summary
    .to_string(index=False)
)


# ============================================================
# 15. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

print("\n" + "=" * 70)
print("СОХРАНЕНИЕ")
print("=" * 70)

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    df.head(1000).to_excel(
        writer,
        sheet_name="sample_raw",
        index=False
    )

    missing.to_excel(
        writer,
        sheet_name="missing",
        index=False
    )

    role_counts.to_excel(
        writer,
        sheet_name="role_frequency",
        index=False
    )

    states.to_excel(
        writer,
        sheet_name="company_year",
        index=False
    )

    transitions.to_excel(
        writer,
        sheet_name="transitions",
        index=False
    )

    largest_changes[
        interesting_columns
    ].head(1000).to_excel(
        writer,
        sheet_name="largest_changes",
        index=False
    )

    company_changes.to_excel(
        writer,
        sheet_name="company_changes",
        index=False
    )

    binary_summary.to_excel(
        writer,
        sheet_name="binary_summary",
        index=False
    )

    delta_summary.to_excel(
        writer,
        sheet_name="delta_summary",
        index=False
    )

print(f"\nРезультат сохранен:")
print(OUTPUT_FILE.resolve())

print("\nГОТОВО.")
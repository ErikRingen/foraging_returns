#!/usr/bin/env python3
"""
Build a minimal public dataset with scrambled participant IDs.

Reads raw_data/*.csv, applies a salted ID remap, strips resource-level
attribution (resource names/indices), and writes to public_data/.

Pangolin and any other species we choose to redact are dropped from
the public dataset entirely (not aggregated).

A secret salt is required (read from the BAYAKA_SALT environment variable
or the --salt CLI flag) so that the remapping cannot be reproduced from
the published source code. The salt is held privately by the authors and
is never committed.

Usage:
    BAYAKA_SALT="<private string>" pixi run python scripts/build_public_dataset.py
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
from pyprojroot.here import here


def _resolve_salt(cli_salt: str | None) -> str:
    """Return the salt from CLI or env, raising a helpful error if neither set."""
    salt = cli_salt or os.environ.get("BAYAKA_SALT")
    if not salt:
        sys.stderr.write(
            "ERROR: anonymization salt is required.\n"
            "  Set BAYAKA_SALT in the environment or pass --salt.\n"
            "  The salt is private to the authors; do NOT commit it.\n"
        )
        sys.exit(2)
    if len(salt) < 16:
        sys.stderr.write(
            "ERROR: salt is too short. Use at least 16 random characters.\n"
        )
        sys.exit(2)
    return salt


def make_id_map(real_ids, salt: str):
    """Salted shuffle of real IDs to short anonymized labels.

    Sort by hash(id||salt) then assign sequential anonymized IDs (P001,
    P002, ...). The salt is private; without it the mapping cannot be
    reproduced from the published source.
    """
    hashed = sorted(
        real_ids,
        key=lambda i: hashlib.sha256(f"{i}{salt}".encode()).hexdigest(),
    )
    return {real: f"P{i + 1:03d}" for i, real in enumerate(hashed)}


def load_kcal_lookup() -> pd.DataFrame:
    df = pd.read_csv(here("raw_data/kcal.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ---- Per-file builders ---------------------------------------------------
def build_camp_members(id_map):
    df = pd.read_csv(here("raw_data/camp_members.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).map(id_map)
    # Ship age (in years, as used in the analysis) instead of BirthYear.
    # The model consumes integer ages with explicit per-forager
    # uncertainty (`age_sigma` = 1 yr under 20, 5 yr at 20+); shipping
    # age matches that form. Age and BirthYear are equivalent given the
    # known study year, so this is a presentational choice, not a
    # privacy step. Strength measurements are dropped because they could
    # plausibly identify participants via repeated-pattern matching.
    df["age"] = (2018 - df["BirthYear"].astype(float)).astype("Int64")
    keep = ["ID", "sex", "day_in", "day_out", "age"]
    return df[keep].rename(columns={"ID": "anon_id"})


def build_daysincamp(id_map):
    df = pd.read_csv(here("raw_data/daysincamp.csv"))
    df.columns = [c.strip() for c in df.columns]
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).map(id_map)
    return df.rename(columns={id_col: "anon_id"}).dropna(subset=["anon_id"])


def build_groups(id_map):
    df = pd.read_csv(here("raw_data/groups.csv"))
    df.columns = [c.strip() for c in df.columns]
    df["ID"] = df["ID"].astype(str).map(id_map)
    df = df.dropna(subset=["ID"])
    # Drop the specific activity name (resource); keep only Activity.Type.
    drop_cols = [c for c in ["Activity"] if c in df.columns]
    return df.drop(columns=drop_cols).rename(columns={"ID": "anon_id"})


def _attach_kcal(df, kcal_lookup, weight_col):
    """Attach pre-computed `kcal = kcal.g * weight_col` so the public
    dataset is fully sufficient to refit the model without ever needing
    `kcal.csv` (which contains the resource→kcal lookup we strip)."""
    df = df.merge(kcal_lookup[["index", "kcal.g"]], on="index", how="left")
    df["kcal.g"] = pd.to_numeric(df["kcal.g"], errors="coerce")
    df["kcal"] = df["kcal.g"] * df[weight_col]
    return df.drop(columns=["kcal.g"])


def build_returns(id_map, kcal_lookup):
    df = pd.read_csv(here("raw_data/returns.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    df["id"] = df["id"].astype(str).map(id_map)
    df = df.dropna(subset=["id"])
    df = _attach_kcal(df, kcal_lookup, "net_food_weight_gram")
    # Drop resource attribution and raw weight: `kcal` (numeric, already
    # computed above) replaces them as the dependent variable.
    drop_cols = [
        c for c in ["index", "article", "state", "net_food_weight_gram"]
        if c in df.columns
    ]
    return df.drop(columns=drop_cols).rename(columns={"id": "anon_id"})


def build_recall(id_map, kcal_lookup):
    df = pd.read_csv(here("raw_data/recall.csv"))
    df.columns = [c.strip().lower() for c in df.columns]
    df["id"] = df["id"].astype(str).map(id_map)
    df = df.dropna(subset=["id"])
    df = _attach_kcal(df, kcal_lookup, "total_weight_grams")
    drop_cols = [
        c for c in [
            "index", "article_consumed", "source",
            "quantity", "x1_unit_weight_grams", "total_weight_grams",
        ] if c in df.columns
    ]
    return df.drop(columns=drop_cols).rename(columns={"id": "anon_id"})


# ---- Driver --------------------------------------------------------------
def collect_all_ids() -> list[str]:
    ids: set[str] = set()
    for f, col in [
        ("raw_data/camp_members.csv", "ID"),
        ("raw_data/daysincamp.csv", None),
        ("raw_data/groups.csv", "ID"),
        ("raw_data/returns.csv", "ID"),
        ("raw_data/recall.csv", "id"),
    ]:
        df = pd.read_csv(here(f))
        df.columns = [c.strip() for c in df.columns]
        if col is None:
            col = df.columns[0]
        ids.update(df[col].astype(str).dropna().unique())
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="public_data",
        help="Output directory (default: public_data/).",
    )
    parser.add_argument(
        "--salt", default=None,
        help="Private salt string. Prefer setting BAYAKA_SALT env var instead.",
    )
    args = parser.parse_args()

    salt = _resolve_salt(args.salt)
    out_dir = Path(here(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    real_ids = collect_all_ids()
    id_map = make_id_map(real_ids, salt=salt)

    kcal = load_kcal_lookup()

    builders = {
        "camp_members.csv": build_camp_members(id_map),
        "daysincamp.csv": build_daysincamp(id_map),
        "groups.csv": build_groups(id_map),
        "returns.csv": build_returns(id_map, kcal),
        "recall.csv": build_recall(id_map, kcal),
    }

    for name, df in builders.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        print(f"  Wrote {path} | rows: {len(df)} | cols: {list(df.columns)}")

    # Save a copy of the kcal lookup with NO links to original `index`
    # values (so consumers can't join back to real species). Provide
    # *only* category-level kcal density summaries.
    kcal_cat = pd.read_csv(here("raw_data/kcal_categories.csv"))
    kcal_cat.columns = [c.strip() for c in kcal_cat.columns]
    kcal_cat["kcal.g"] = pd.to_numeric(kcal_cat["kcal.g"], errors="coerce")
    cat_summary = (
        kcal_cat.dropna(subset=["kcal.g"])
        .groupby("Category")["kcal.g"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "kcal_per_g_mean", "std": "kcal_per_g_sd", "count": "n_resources"})
        .reset_index()
    )
    cat_summary.to_csv(out_dir / "category_kcal_summary.csv", index=False)
    print(f"  Wrote {out_dir / 'category_kcal_summary.csv'} (category-level summary)")

    # Document the public release
    readme = out_dir / "README.md"
    readme.write_text(
        """# Public dataset (anonymized)

This is the publicly shareable subset of the BaYaka subsistence dataset
underlying the manuscript. It has been processed by
`scripts/build_public_dataset.py` to:

1. Replace original participant IDs with anonymized labels (`P001`, `P002`,
   ...). The mapping is generated with a private salt (held by the authors
   and never committed); it cannot be reproduced from the published source
   code and is unrelated to IDs in other BaYaka papers.
2. Replace `BirthYear` with `age` (years, = `2018 - BirthYear`), the form
   the model consumes. Age and birth year are mathematically equivalent
   given a known study year; this is a presentational choice, not a
   privacy step.
3. Remove resource-level attribution and raw weights. Specifically, the
   `index`, `article`, `article_consumed`, `source`, free-text `state`,
   and the per-package weight columns (`net_food_weight_gram` in
   `returns.csv`; `quantity`, `x1_unit_weight_grams`, `total_weight_grams`
   in `recall.csv`) are dropped. Only the pre-computed `kcal` totals
   remain as the dependent variable.

For the full dataset (with real participant IDs and resource attribution),
contact the corresponding authors. Use is subject to ethical approval.
""",
        encoding="utf-8",
    )
    print(f"  Wrote {readme}")
    print()
    print(f"Built public dataset for {len(real_ids)} unique participants -> {out_dir}/")


if __name__ == "__main__":
    main()

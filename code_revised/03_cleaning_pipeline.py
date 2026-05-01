from datetime import datetime
import runpy
from pathlib import Path

import pandas as pd


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))


def convert_endtime(value):
    return datetime.strptime(str(value).split()[0], "%d/%m/%Y")


def apply_missing_threshold(df):
    missing_table = pd.read_csv(cfg["DATA_DIR"] / "missing_value_counts.csv")
    drop_cols = missing_table.loc[
        missing_table["Missing Value Count"] > cfg["MISSING_THRESHOLD"],
        "Variable Name",
    ].tolist()
    return df.drop(columns=drop_cols), drop_cols


def encode_structural_health_missingness(df):
    mask = (df["endtime"] >= pd.Timestamp("2021-02-10")) & (
        df["endtime"] <= pd.Timestamp("2021-10-18")
    )
    for col in cfg["PHQ4_ITEMS"] + cfg["COMORBIDITY_ITEMS"]:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col].fillna("N/A")
    return df


def recode_original_scales(df):
    for col in ["r1_1", "r1_2"]:
        if col in df.columns:
            df[col] = df[col].map(lambda value: cfg["AGREEMENT_MAP"].get(value, value))

    for col in [c for c in df.columns if c.startswith("i12_health_")]:
        df[col] = df[col].map(cfg["FREQUENCY_MAP"])

    return df


def main():
    cfg["ensure_output_dirs"]()

    df = pd.read_csv(
        cfg["YOUGOV_PATH"],
        na_values=[" ", "__NA__"],
        keep_default_na=True,
        low_memory=False,
    )
    df["endtime"] = df["endtime"].apply(convert_endtime)
    df, dropped_cols = apply_missing_threshold(df)
    df = encode_structural_health_missingness(df)
    df = df.dropna()
    df = recode_original_scales(df)

    df.to_csv(cfg["DATA_DIR"] / "cleaned_base_before_features.csv", index=False)
    pd.Series(dropped_cols, name="dropped_variable").to_csv(
        cfg["DATA_DIR"] / "dropped_missingness_variables.csv",
        index=False,
    )

    print("Base cleaned shape before feature engineering:", df.shape)
    print("Dropped columns above threshold:", len(dropped_cols))


if __name__ == "__main__":
    main()


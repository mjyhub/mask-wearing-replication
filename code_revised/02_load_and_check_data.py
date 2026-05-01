import runpy
from pathlib import Path

import pandas as pd


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))


def read_yougov_raw():
    return pd.read_csv(
        cfg["YOUGOV_PATH"],
        na_values=[" ", "__NA__"],
        keep_default_na=True,
        low_memory=False,
    )


def make_missing_table(df):
    return (
        df.isna()
        .sum()
        .rename("Missing Value Count")
        .reset_index()
        .rename(columns={"index": "Variable Name"})
        .sort_values(["Missing Value Count", "Variable Name"])
    )


def make_mandate_start_dates():
    policy = pd.read_csv(
        cfg["POLICY_PATH"],
        usecols=["RegionName", "RegionCode", "Date", "H6M_Facial Coverings"],
    )
    policy["Date"] = pd.to_datetime(policy["Date"], format="%Y%m%d")
    policy = policy.sort_values(["RegionName", "Date"])

    policy["h6m_rolling_mean"] = (
        policy.groupby("RegionName")["H6M_Facial Coverings"]
        .rolling(window=cfg["MANDATE_ROLLING_DAYS"], min_periods=cfg["MANDATE_ROLLING_DAYS"])
        .mean()
        .reset_index(level=0, drop=True)
    )

    mandates = (
        policy.loc[policy["h6m_rolling_mean"] >= cfg["MANDATE_THRESHOLD"]]
        .groupby("RegionName", as_index=False)
        .first()
    )
    mandates = mandates[
        ["RegionName", "RegionCode", "Date", "H6M_Facial Coverings", "h6m_rolling_mean"]
    ]
    mandates["Date"] = mandates["Date"].dt.strftime("%Y-%m-%d")
    return mandates


def main():
    cfg["ensure_output_dirs"]()

    yougov = read_yougov_raw()
    missing_table = make_missing_table(yougov)
    mandate_dates = make_mandate_start_dates()

    missing_table.to_csv(cfg["DATA_DIR"] / "missing_value_counts.csv", index=False)
    mandate_dates.to_csv(cfg["DATA_DIR"] / "mandate_start_dates.csv", index=False)

    print("YouGov raw shape:", yougov.shape)
    print("Missing table saved:", cfg["DATA_DIR"] / "missing_value_counts.csv")
    print("Mandate dates saved:", cfg["DATA_DIR"] / "mandate_start_dates.csv")


if __name__ == "__main__":
    main()


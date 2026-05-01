import runpy
from pathlib import Path

import pandas as pd


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))


def add_mandate_period(df, mandate_dates):
    lookup = mandate_dates.set_index("RegionName")["Date"].to_dict()
    lookup = {state: pd.to_datetime(date) for state, date in lookup.items()}

    df["endtime"] = pd.to_datetime(df["endtime"])
    df["within_mandate_period"] = df.apply(
        lambda row: int(row["endtime"] >= lookup[row["state"]]),
        axis=1,
    )
    return df


def dummy_encode(df):
    encoded = df.copy()
    for col in cfg["DUMMY_COLUMNS"]:
        dummy = pd.get_dummies(encoded[col], prefix=col, drop_first=True)
        encoded = pd.concat([encoded.drop(columns=col), dummy], axis=1)
    return encoded


def main():
    cfg["ensure_output_dirs"]()

    cleaned = pd.read_csv(cfg["DATA_DIR"] / "cleaned_data.csv", keep_default_na=False)
    mandates = pd.read_csv(cfg["DATA_DIR"] / "mandate_start_dates.csv")

    with_period = add_mandate_period(cleaned, mandates)
    preprocessed = dummy_encode(with_period)
    preprocessed.to_csv(cfg["DATA_DIR"] / "cleaned_data_preprocessing.csv", index=False)

    print("Preprocessed data shape:", preprocessed.shape)
    print(preprocessed["within_mandate_period"].value_counts().sort_index())


if __name__ == "__main__":
    main()


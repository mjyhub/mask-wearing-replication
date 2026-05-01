import runpy
from pathlib import Path

import pandas as pd


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))


def household_convert(value):
    if value in [str(i) for i in range(1, 8)]:
        return int(value)
    if value == "8 or more":
        return 8
    if value in ["Prefer not to say", "Don't know"]:
        return None
    return value


def construct_behaviour_variables(df):
    protective_cols = [col for col in df.columns if col.startswith("i12_")]
    nomask_cols = [col for col in protective_cols if col not in cfg["FACE_MASK_ITEMS"]]

    df["face_mask_behaviour_scale"] = df[cfg["FACE_MASK_ITEMS"]].median(axis=1)
    df["face_mask_behaviour_binary"] = df["face_mask_behaviour_scale"].apply(
        lambda value: "Yes" if value >= 4 else "No"
    )

    df["protective_behaviour_scale"] = df[protective_cols].median(axis=1)
    df["protective_behaviour_binary"] = df["protective_behaviour_scale"].apply(
        lambda value: "Yes" if value >= 4 else "No"
    )

    df["protective_behaviour_nomask_scale"] = df[nomask_cols].median(axis=1)
    return df, protective_cols


def construct_comorbidities(df):
    d1_cols = [col for col in df.columns if col.startswith("d1_")]
    df["d1_comorbidities"] = "Yes"
    df.loc[df["d1_health_99"] == "Yes", "d1_comorbidities"] = "No"
    df.loc[df["d1_health_99"] == "N/A", "d1_comorbidities"] = "NA"
    df.loc[df["d1_health_98"] == "Yes", "d1_comorbidities"] = "Prefer_not_to_say"
    return df.drop(columns=d1_cols)


def main():
    cfg["ensure_output_dirs"]()

    df = pd.read_csv(
        cfg["DATA_DIR"] / "cleaned_base_before_features.csv",
        keep_default_na=False,
    )
    df["endtime"] = pd.to_datetime(df["endtime"])

    df, protective_cols = construct_behaviour_variables(df)
    df = construct_comorbidities(df)

    start_date = df["endtime"].min()
    df["week_number"] = ((df["endtime"] - start_date).dt.days // 14) + 1

    df["household_size"] = df["household_size"].apply(household_convert)
    df = df.dropna()

    drop_cols = ["qweek", "weight"] + protective_cols
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    df.to_csv(cfg["DATA_DIR"] / "cleaned_data.csv", index=False)
    print("Cleaned data shape:", df.shape)


if __name__ == "__main__":
    main()


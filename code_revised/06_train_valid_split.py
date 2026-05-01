import runpy
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


cfg = runpy.run_path(Path(__file__).with_name("01_config.py"))


def subset_period(df, period):
    if period == "all":
        return df
    if period == "before":
        return df.loc[df["within_mandate_period"] == 0]
    if period == "after":
        return df.loc[df["within_mandate_period"] == 1]
    raise ValueError(f"Unknown period: {period}")


def write_model_dataset(df_train, df_test, model_name, spec):
    train_sub = subset_period(df_train, spec["period"])
    test_sub = subset_period(df_test, spec["period"])
    feature_cols = train_sub.columns.drop(spec["drop"])

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_sub[spec["target"]].values.ravel())
    y_test = encoder.transform(test_sub[spec["target"]].values.ravel())

    train_sub[feature_cols].to_csv(cfg["DATA_DIR"] / f"X_train_{model_name}.csv", index=False)
    test_sub[feature_cols].to_csv(cfg["DATA_DIR"] / f"X_test_{model_name}.csv", index=False)
    pd.DataFrame({"y_train": y_train}).to_csv(
        cfg["DATA_DIR"] / f"y_train_{model_name}.csv",
        index=False,
    )
    pd.DataFrame({"y_test": y_test}).to_csv(
        cfg["DATA_DIR"] / f"y_test_{model_name}.csv",
        index=False,
    )

    return {
        "model": model_name,
        "target": spec["target"],
        "period": spec["period"],
        "train_rows": len(train_sub),
        "test_rows": len(test_sub),
        "feature_count": len(feature_cols),
        "target_classes": "|".join(encoder.classes_),
    }


def write_audit(cleaned, preprocessed, df_train, df_test, model_summary):
    raw = pd.read_csv(
        cfg["YOUGOV_PATH"],
        na_values=[" ", "__NA__"],
        keep_default_na=True,
        low_memory=False,
    )
    audit = pd.DataFrame(
        [
            {
                "raw_rows": raw.shape[0],
                "raw_columns": raw.shape[1],
                "cleaned_rows": cleaned.shape[0],
                "cleaned_columns": cleaned.shape[1],
                "preprocessed_rows": preprocessed.shape[0],
                "preprocessed_columns": preprocessed.shape[1],
                "before_mandate_rows": int((preprocessed["within_mandate_period"] == 0).sum()),
                "after_mandate_rows": int((preprocessed["within_mandate_period"] == 1).sum()),
                "train_rows": df_train.shape[0],
                "test_rows": df_test.shape[0],
            }
        ]
    )
    audit.to_csv(cfg["RESULTS_DIR"] / "pipeline_audit_pipeline_shapes.csv", index=False)
    model_summary.to_csv(cfg["RESULTS_DIR"] / "pipeline_audit_model_datasets.csv", index=False)
    return audit


def main():
    cfg["ensure_output_dirs"]()

    cleaned = pd.read_csv(cfg["DATA_DIR"] / "cleaned_data.csv", keep_default_na=False)
    preprocessed = pd.read_csv(
        cfg["DATA_DIR"] / "cleaned_data_preprocessing.csv",
        keep_default_na=False,
    )

    df_train, df_test = train_test_split(
        preprocessed,
        test_size=cfg["TEST_SIZE"],
        random_state=cfg["TRAIN_TEST_SEED"],
        stratify=preprocessed["within_mandate_period"],
    )
    df_train.to_csv(cfg["DATA_DIR"] / "df_train.csv", index=False)
    df_test.to_csv(cfg["DATA_DIR"] / "df_test.csv", index=False)

    rows = [
        write_model_dataset(df_train, df_test, model_name, spec)
        for model_name, spec in cfg["MODEL_SPECS"].items()
    ]
    model_summary = pd.DataFrame(rows)
    audit = write_audit(cleaned, preprocessed, df_train, df_test, model_summary)

    print(audit.to_string(index=False))
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()


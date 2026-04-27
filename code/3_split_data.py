import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

cleaned_df = pd.read_csv(
    "../data/cleaned_data_preprocessing.csv", keep_default_na=False)
# feature_cols = cleaned_df.columns.drop(
#     ["RecordNo", "face_mask_behaviour_scale", "face_mask_behaviour_binary", "endtime"])
# for col in feature_cols:
#     print(col)

# Label encoder
# label_encoder = LabelEncoder()
# %%

mandate_period_label = cleaned_df.loc[:, "within_mandate_period"]

df_train, df_test = train_test_split(cleaned_df,
                                     test_size=0.2,
                                     random_state=20240417,
                                     stratify=mandate_period_label)

df_train.to_csv("../data/df_train.csv", index=False)
df_test.to_csv("../data/df_test.csv", index=False)

# %% Model 1: Predicting face masks

label_encoder = LabelEncoder()

response_col = ["face_mask_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "endtime"])

X_train_model_1 = df_train[feature_cols]
X_test_model_1 = df_test[feature_cols]

y_train_model_1 = label_encoder.fit_transform(
    df_train[response_col].values.ravel())
y_test_model_1 = label_encoder.fit_transform(
    df_test[response_col].values.ravel())

X_train_model_1.to_csv("../data/X_train_model_1.csv", index=False)
X_test_model_1.to_csv("../data/X_test_model_1.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1}).to_csv(
    "../data/y_train_model_1.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1}).to_csv(
    "../data/y_test_model_1.csv", index=False)

# %% Model 1a: Predicting face masks in early time

mandate_starter = "2022-01-01"


response_col = ["face_mask_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = (df_train["endtime"] < mandate_starter) & (
    df_train["within_mandate_period"] == 0)
logic_subsetter_test = (df_test["endtime"] < mandate_starter) & (
    df_test["within_mandate_period"] == 0)

X_train_model_1a = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_1a = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_1a = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_1a = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_1a.to_csv("../data/X_train_model_1a.csv", index=False)
X_test_model_1a.to_csv("../data/X_test_model_1a.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1a}).to_csv(
    "../data/y_train_model_1a.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1a}).to_csv(
    "../data/y_test_model_1a.csv", index=False)

# %% Model 1b: Predicting face masks in mandate periods


response_col = ["face_mask_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = df_train["within_mandate_period"] == 1
logic_subsetter_test = df_test["within_mandate_period"] == 1

X_train_model_1b = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_1b = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_1b = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_1b = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_1b.to_csv("../data/X_train_model_1b.csv", index=False)
X_test_model_1b.to_csv("../data/X_test_model_1b.csv", index=False)
pd.DataFrame({'y_train': y_train_model_1b}).to_csv(
    "../data/y_train_model_1b.csv", index=False)
pd.DataFrame({'y_test': y_test_model_1b}).to_csv(
    "../data/y_test_model_1b.csv", index=False)


# %% Model 2: Predicting protective behaviour

label_encoder = LabelEncoder()

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime"])

X_train_model_2 = df_train[feature_cols]
X_test_model_2 = df_test[feature_cols]

y_train_model_2 = label_encoder.fit_transform(
    df_train[response_col].values.ravel())
y_test_model_2 = label_encoder.fit_transform(
    df_test[response_col].values.ravel())

X_train_model_2.to_csv("../data/X_train_model_2.csv", index=False)
X_test_model_2.to_csv("../data/X_test_model_2.csv", index=False)
pd.DataFrame({'y_train': y_train_model_2}).to_csv(
    "../data/y_train_model_2.csv", index=False)
pd.DataFrame({'y_test': y_test_model_2}).to_csv(
    "../data/y_test_model_2.csv", index=False)

# %% Model 2a: Predicting protective behaviour in early time


mandate_starter = "2022-01-01"


response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = (df_train["endtime"] < mandate_starter) & (
    df_train["within_mandate_period"] == 0)
logic_subsetter_test = (df_test["endtime"] < mandate_starter) & (
    df_test["within_mandate_period"] == 0)

X_train_model_2a = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_2a = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_2a = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_2a = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_2a.to_csv("../data/X_train_model_2a.csv", index=False)
X_test_model_2a.to_csv("../data/X_test_model_2a.csv", index=False)
pd.DataFrame({'y_train': y_train_model_2a}).to_csv(
    "../data/y_train_model_2a.csv", index=False)
pd.DataFrame({'y_test': y_test_model_2a}).to_csv(
    "../data/y_test_model_2a.csv", index=False)

# %% Model 2b: Predicting protective behaviour in mandate periods

response_col = ["protective_behaviour_binary"]

feature_cols = cleaned_df.columns.drop(["RecordNo",
                                        "face_mask_behaviour_scale",
                                        "protective_behaviour_scale",
                                        "face_mask_behaviour_binary",
                                        "protective_behaviour_binary",
                                        "protective_behaviour_nomask_scale",
                                        "endtime",
                                        "within_mandate_period"])

logic_subsetter_train = df_train["within_mandate_period"] == 1
logic_subsetter_test = df_test["within_mandate_period"] == 1

X_train_model_2b = df_train.loc[logic_subsetter_train, feature_cols]
X_test_model_2b = df_test.loc[logic_subsetter_test, feature_cols]

y_train_model_2b = label_encoder.fit_transform(
    df_train.loc[logic_subsetter_train, response_col].values.ravel())
y_test_model_2b = label_encoder.fit_transform(
    df_test.loc[logic_subsetter_test, response_col].values.ravel())

X_train_model_2b.to_csv("../data/X_train_model_2b.csv", index=False)
X_test_model_2b.to_csv("../data/X_test_model_2b.csv", index=False)
pd.DataFrame({'y_train': y_train_model_2b}).to_csv(
    "../data/y_train_model_2b.csv", index=False)
pd.DataFrame({'y_test': y_test_model_2b}).to_csv(
    "../data/y_test_model_2b.csv", index=False)

###########################
# %% Summary of generated datasets

print("\n===== Unified train/test split =====")
print("df_train shape:", df_train.shape)
print("df_test shape:", df_test.shape)


shape_summary = pd.DataFrame({
    "Dataset": [
        "df_train",
        "df_test",
        "X_train_model_1",
        "X_test_model_1",
        "X_train_model_1a",
        "X_test_model_1a",
        "X_train_model_1b",
        "X_test_model_1b",
        "X_train_model_2",
        "X_test_model_2",
        "X_train_model_2a",
        "X_test_model_2a",
        "X_train_model_2b",
        "X_test_model_2b"
    ],
    "Rows": [
        df_train.shape[0],
        df_test.shape[0],
        X_train_model_1.shape[0],
        X_test_model_1.shape[0],
        X_train_model_1a.shape[0],
        X_test_model_1a.shape[0],
        X_train_model_1b.shape[0],
        X_test_model_1b.shape[0],
        X_train_model_2.shape[0],
        X_test_model_2.shape[0],
        X_train_model_2a.shape[0],
        X_test_model_2a.shape[0],
        X_train_model_2b.shape[0],
        X_test_model_2b.shape[0]
    ],
    "Columns": [
        df_train.shape[1],
        df_test.shape[1],
        X_train_model_1.shape[1],
        X_test_model_1.shape[1],
        X_train_model_1a.shape[1],
        X_test_model_1a.shape[1],
        X_train_model_1b.shape[1],
        X_test_model_1b.shape[1],
        X_train_model_2.shape[1],
        X_test_model_2.shape[1],
        X_train_model_2a.shape[1],
        X_test_model_2a.shape[1],
        X_train_model_2b.shape[1],
        X_test_model_2b.shape[1]
    ]
})

print("\n===== Dataset shape summary =====")
print(shape_summary)

shape_summary.to_csv("../data/model_dataset_shape_summary.csv", index=False)


def class_distribution(y, dataset_name):
    counts = pd.Series(y).value_counts().sort_index()
    total = counts.sum()

    return pd.DataFrame({
        "Dataset": dataset_name,
        "Class": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / total * 100).round(2)
    })


class_summary = pd.concat([
    class_distribution(y_train_model_1, "y_train_model_1"),
    class_distribution(y_test_model_1, "y_test_model_1"),

    class_distribution(y_train_model_1a, "y_train_model_1a"),
    class_distribution(y_test_model_1a, "y_test_model_1a"),

    class_distribution(y_train_model_1b, "y_train_model_1b"),
    class_distribution(y_test_model_1b, "y_test_model_1b"),

    class_distribution(y_train_model_2, "y_train_model_2"),
    class_distribution(y_test_model_2, "y_test_model_2"),

    class_distribution(y_train_model_2a, "y_train_model_2a"),
    class_distribution(y_test_model_2a, "y_test_model_2a"),

    class_distribution(y_train_model_2b, "y_train_model_2b"),
    class_distribution(y_test_model_2b, "y_test_model_2b")
], ignore_index=True)

print("\n===== Y class distribution summary =====")
print(class_summary)

class_summary.to_csv("../data/model_y_class_distribution_summary.csv", index=False)
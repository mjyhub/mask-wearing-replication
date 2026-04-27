##########用 pkl 文件生成模型表现汇总表
import pickle
import pandas as pd
import numpy as np

model_list = [
    "model_1",
    "model_1a",
    "model_1b",
    "model_2",
    "model_2a",
    "model_2b"
]

model_descriptions = {
    "model_1": "Face mask behaviour - full sample",
    "model_1a": "Face mask behaviour - early non-mandate period",
    "model_1b": "Face mask behaviour - mandate period",
    "model_2": "Protective behaviour - full sample",
    "model_2a": "Protective behaviour - early non-mandate period",
    "model_2b": "Protective behaviour - mandate period"
}

rows = []

for model in model_list:
    file_path = f"../results/{model}_logistic_reg.pkl"

    with open(file_path, "rb") as f:
        cv_scores = pickle.load(f)

    row = {
        "Model": model,
        "Description": model_descriptions[model],
        "Precision_mean": np.mean(cv_scores["test_precision"]),
        "Precision_sd": np.std(cv_scores["test_precision"]),
        "Recall_mean": np.mean(cv_scores["test_recall"]),
        "Recall_sd": np.std(cv_scores["test_recall"]),
        "ROC_AUC_mean": np.mean(cv_scores["test_roc_auc"]),
        "ROC_AUC_sd": np.std(cv_scores["test_roc_auc"]),
        "Accuracy_mean": np.mean(cv_scores["test_accuracy"]),
        "Accuracy_sd": np.std(cv_scores["test_accuracy"]),
        "F1_mean": np.mean(cv_scores["test_f1"]),
        "F1_sd": np.std(cv_scores["test_f1"])
    }

    rows.append(row)

summary_df = pd.DataFrame(rows)

# Round values
numeric_cols = summary_df.select_dtypes(include="number").columns
summary_df[numeric_cols] = summary_df[numeric_cols].round(3)

print(summary_df)

summary_df.to_csv("../results/logistic_regression_summary.csv", index=False)


#########画模型表现对比图
import matplotlib.pyplot as plt

# Read summary table
summary_df = pd.read_csv("../results/logistic_regression_summary.csv")

# Set model names
models = summary_df["Model"]

# Plot Recall
plt.figure(figsize=(8, 5))
plt.bar(models, summary_df["Recall_mean"])
plt.xlabel("Model")
plt.ylabel("Mean Recall")
plt.title("Logistic Regression Mean Recall by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_regression_recall.png", dpi=300)
plt.show()

# Plot ROC-AUC
plt.figure(figsize=(8, 5))
plt.bar(models, summary_df["ROC_AUC_mean"])
plt.xlabel("Model")
plt.ylabel("Mean ROC-AUC")
plt.title("Logistic Regression Mean ROC-AUC by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_regression_roc_auc.png", dpi=300)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.bar(models, summary_df["Accuracy_mean"])
plt.xlabel("Model")
plt.ylabel("Mean Accuracy")
plt.title("Logistic Regression Mean Accuracy by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_regression_accuracy.png", dpi=300)
plt.show()

# Plot F1-score
plt.figure(figsize=(8, 5))
plt.bar(models, summary_df["F1_mean"])
plt.xlabel("Model")
plt.ylabel("Mean F1-score")
plt.title("Logistic Regression Mean F1-score by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/logistic_regression_f1.png", dpi=300)
plt.show()

################画一个综合对比图
import pandas as pd
import matplotlib.pyplot as plt

summary_df = pd.read_csv("../results/logistic_regression_summary.csv")

plot_df = summary_df[[
    "Model",
    "Recall_mean",
    "ROC_AUC_mean",
    "Accuracy_mean",
    "F1_mean"
]]

plot_df = plot_df.rename(columns={
    "Recall_mean": "Recall",
    "ROC_AUC_mean": "ROC-AUC",
    "Accuracy_mean": "Accuracy",
    "F1_mean": "F1-score"
})

plot_df.set_index("Model").plot(kind="bar", figsize=(10, 6))

plt.xlabel("Model")
plt.ylabel("Mean score")
plt.title("Logistic Regression Cross-validation Performance")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("../results/logistic_regression_all_metrics.png", dpi=300)
plt.show()

##########make_logistic_table简化版
import pandas as pd

# Read the full summary table

df = pd.read_csv("../results/logistic_regression_summary.csv")

# Keep only key columns for the report

report_table = df[[

    "Model",

    "Description",

    "Precision_mean",

    "Recall_mean",

    "ROC_AUC_mean",

    "Accuracy_mean",

    "F1_mean"

]].copy()

# Rename columns for the paper

report_table = report_table.rename(columns={

    "Description": "Prediction task",

    "Precision_mean": "Precision",

    "Recall_mean": "Recall",

    "ROC_AUC_mean": "ROC-AUC",

    "Accuracy_mean": "Accuracy",

    "F1_mean": "F1-score"

})

# Round numbers to 3 decimal places

numeric_cols = ["Precision", "Recall", "ROC-AUC", "Accuracy", "F1-score"]

report_table[numeric_cols] = report_table[numeric_cols].round(3)

print(report_table)

# Save simplified table

report_table.to_csv("../results/logistic_regression_report_table.csv", index=False)
#!/usr/bin/env Rscript

library(dplyr)
library(gt)
library(readr)

project_dir <- normalizePath(getwd(), mustWork = FALSE)
results_dir <- file.path(project_dir, "results")
figures_dir <- file.path(project_dir, "figures")

dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

model_labels <- c(
  logistic_reg = "logistic regression",
  binary_tree = "classification tree",
  xgboost = "XGBoost",
  rf = "random forest"
)

format_mean_se <- function(mean_value, se_value) {
  sprintf("%.3f (%.3f)", mean_value, se_value)
}

style_table <- function(tab) {
  tab %>%
    tab_options(
      table.font.names = c("system-ui", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"),
      table.font.size = px(14),
      heading.title.font.size = px(16),
      heading.align = "center",
      table.border.top.color = "#A8A8A8",
      table.border.top.width = px(2),
      table.border.bottom.color = "#A8A8A8",
      table.border.bottom.width = px(2),
      column_labels.border.top.color = "#D3D3D3",
      column_labels.border.top.width = px(2),
      column_labels.border.bottom.color = "#D3D3D3",
      column_labels.border.bottom.width = px(2),
      row_group.border.top.color = "#D3D3D3",
      row_group.border.top.width = px(2),
      row_group.border.bottom.color = "#D3D3D3",
      row_group.border.bottom.width = px(2),
      data_row.padding = px(8),
      row_group.padding = px(8)
    ) %>%
    tab_style(
      style = cell_text(weight = "normal"),
      locations = cells_column_labels(everything())
    ) %>%
    tab_style(
      style = cell_text(weight = "normal"),
      locations = cells_row_groups()
    )
}

read_cv_file <- function(file_name, group_label) {
  read_csv(file.path(results_dir, file_name), show_col_types = FALSE) %>%
    mutate(
      group = group_label,
      model_type = recode(model_type, !!!model_labels),
      AUC = format_mean_se(roc_auc, roc_auc_std),
      precision = format_mean_se(precision, precision_std),
      recall = format_mean_se(recall, recall_std),
      accuracy = format_mean_se(accuracy, accuracy_std),
      F1 = format_mean_se(f1, f1_std)
    ) %>%
    select(group, model_type, AUC, precision, recall, accuracy, F1)
}

# =========================
# Table 1: CV results
# =========================

table1 <- bind_rows(
  read_cv_file("03_model_1a_final_results.csv", "before mandates—face mask wearing"),
  read_cv_file("05_model_1b_final_results.csv", "after mandates—face mask wearing"),
  read_cv_file("04_model_2a_final_results.csv", "before mandates—general protective behaviour"),
  read_cv_file("06_model_2b_final_results.csv", "after mandates—general protective behaviour")
) %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "before mandates—face mask wearing",
        "after mandates—face mask wearing",
        "before mandates—general protective behaviour",
        "after mandates—general protective behaviour"
      )
    ),
    model_type = factor(
      model_type,
      levels = c("logistic regression", "classification tree", "XGBoost", "random forest")
    )
  ) %>%
  arrange(group, model_type)

table1_gt <- table1 %>%
  gt(rowname_col = "model_type", groupname_col = "group") %>%
  tab_header(
    title = md("**Table 1.** Fivefold cross-validation results comparing four predictive models")
  ) %>%
  cols_label(
    AUC = "AUC",
    precision = "precision",
    recall = "recall",
    accuracy = "accuracy",
    F1 = "F1"
  ) %>%
  style_table()

gtsave(
  table1_gt,
  filename = file.path(figures_dir, "Table_1_cv_results.html")
)

# =========================
# Table 2: validation results
# =========================

table2_lookup <- tibble(
  model_number = c("model_1a", "model_2a", "model_1b", "model_2b"),
  group = c(
    "before mandates—face mask wearing",
    "before mandates—general protective behaviour",
    "after mandates—face mask wearing",
    "after mandates—general protective behaviour"
  )
)

table2 <- read_csv(
  file.path(results_dir, "final_model_test_metrics.csv"),
  show_col_types = FALSE
) %>%
  inner_join(table2_lookup, by = "model_number") %>%
  filter(model_type %in% c("xgboost", "rf")) %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "before mandates—face mask wearing",
        "before mandates—general protective behaviour",
        "after mandates—face mask wearing",
        "after mandates—general protective behaviour"
      )
    ),
    model_type = recode(model_type, !!!model_labels),
    model_type = factor(model_type, levels = c("XGBoost", "random forest")),
    AUC = sprintf("%.3f", test_roc_auc),
    precision = sprintf("%.3f", test_precision),
    recall = sprintf("%.3f", test_recall),
    accuracy = sprintf("%.3f", test_accuracy),
    F1 = sprintf("%.3f", test_f1)
  ) %>%
  arrange(group, model_type) %>%
  select(group, model_type, AUC, precision, recall, accuracy, F1)

table2_gt <- table2 %>%
  gt(rowname_col = "model_type", groupname_col = "group") %>%
  tab_header(
    title = md("**Table 2.** Metric evaluation on an independent validation set for the optimal models")
  ) %>%
  cols_label(
    AUC = "AUC",
    precision = "precision",
    recall = "recall",
    accuracy = "accuracy",
    F1 = "F1"
  ) %>%
  style_table()

gtsave(
  table2_gt,
  filename = file.path(figures_dir, "Table_2_validation_results.html")
)

print("Saved Table_1_cv_results.html")
print("Saved Table_2_validation_results.html")


library(tidyverse)
library(here)
library(glue)
library(gt)

# =========================
# Table 1: CV results
# =========================

read_cv_file <- function(path, section_name){
  read_csv(path, col_types = cols()) %>%
    mutate(section = section_name)
}

t1_1a <- read_cv_file(
  here("results/03_model_1a_final_results.csv"),
  "before mandates—face mask wearing"
)

t1_1b <- read_cv_file(
  here("results/05_model_1b_final_results.csv"),
  "after mandates—face mask wearing"
)

t1_2a <- read_cv_file(
  here("results/04_model_2a_final_results.csv"),
  "before mandates—general protective behaviour"
)

t1_2b <- read_cv_file(
  here("results/06_model_2b_final_results.csv"),
  "after mandates—general protective behaviour"
)

table1_df <- bind_rows(t1_1a, t1_1b, t1_2a, t1_2b) %>%
  mutate(
    model_type = recode(
      model_type,
      "logistic_reg" = "logistic regression",
      "binary_tree" = "classification tree",
      "xgboost" = "XGBoost",
      "rf" = "random forest"
    ),
    AUC = sprintf("%.3f (%.3f)", roc_auc, roc_auc_std),
    precision = sprintf("%.3f (%.3f)", precision, precision_std),
    recall = sprintf("%.3f (%.3f)", recall, recall_std),
    accuracy = sprintf("%.3f (%.3f)", accuracy, accuracy_std),
    F1 = sprintf("%.3f (%.3f)", f1, f1_std)
  ) %>%
  select(section, model_type, AUC, precision, recall, accuracy, F1)

write_csv(table1_df, here("results/Table_1_cv_results_formatted.csv"))

# gt version
table1_gt <- table1_df %>%
  gt(groupname_col = "section") %>%
  tab_header(
    title = md("**Table 1.** Fivefold cross-validation results comparing four predictive models")
  ) %>%
  cols_label(
    model_type = "",
    AUC = "AUC",
    precision = "precision",
    recall = "recall",
    accuracy = "accuracy",
    F1 = "F1"
  ) %>%
  tab_options(
    table.font.size = 14,
    heading.title.font.size = 16
  )

gtsave(table1_gt, here("figures/Table_1_cv_results.html"))

# =========================
# Table 2: test / validation results
# =========================

table2_df <- read_csv(here("results/final_model_test_metrics.csv"), col_types = cols()) %>%
  filter(
    model_number %in% c("model_1a", "model_1b", "model_2a", "model_2b"),
    model_type %in% c("xgboost", "rf")
  ) %>%
  mutate(
    section = case_when(
      model_number == "model_1a" ~ "before mandates—face mask wearing",
      model_number == "model_1b" ~ "after mandates—face mask wearing",
      model_number == "model_2a" ~ "before mandates—general protective behaviour",
      model_number == "model_2b" ~ "after mandates—general protective behaviour"
    ),
    model_type = recode(
      model_type,
      "xgboost" = "XGBoost",
      "rf" = "random forest"
    ),
    AUC = sprintf("%.3f", test_roc_auc),
    precision = sprintf("%.3f", test_precision),
    recall = sprintf("%.3f", test_recall),
    accuracy = sprintf("%.3f", test_accuracy),
    F1 = sprintf("%.3f", test_f1)
  ) %>%
  select(section, model_type, AUC, precision, recall, accuracy, F1)

write_csv(table2_df, here("results/Table_2_validation_results_formatted.csv"))

table2_gt <- table2_df %>%
  gt(groupname_col = "section") %>%
  tab_header(
    title = md("**Table 2.** Metric evaluation on an independent validation set for the optimal models")
  ) %>%
  cols_label(
    model_type = "",
    AUC = "AUC",
    precision = "precision",
    recall = "recall",
    accuracy = "accuracy",
    F1 = "F1"
  ) %>%
  tab_options(
    table.font.size = 14,
    heading.title.font.size = 16
  )

gtsave(table2_gt, here("figures/Table_2_validation_results.html"))
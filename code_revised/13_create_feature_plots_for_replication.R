## Replication feature importance plots
## Adapted for your repeated feature importance files
## Use in RStudio

# libraries ---------------------------------------------------------------
library(tidyverse)
library(patchwork)
library(here)
library(glue)
library(RColorBrewer)

# script parameters -------------------------------------------------------
text_size <- 18
height <- 15
width <- 14
dpi <- 600

# functions ---------------------------------------------------------------

# 读取 repeated csv，并按 Ryan 思路算 median / q1 / q3
load_data <- function(model_type, model_number){
  
  # 这里读取的是你 Python 生成的 repeated 文件
  df1 <- read_csv(
    here(glue("results/{model_number}a_{model_type}_feature_importance_repeated.csv")),
    col_types = cols()
  )
  
  df2 <- read_csv(
    here(glue("results/{model_number}b_{model_type}_feature_importance_repeated.csv")),
    col_types = cols()
  )
  
  # 去掉 run_id 和 seed，只保留 feature importance 列
  df1 <- df1 %>% select(-run_id, -seed)
  df2 <- df2 %>% select(-run_id, -seed)
  
  # before mandates
  df_pre_long <- df1 %>%
    pivot_longer(everything()) %>%
    group_by(name) %>%
    summarise(
      median_importance = median(value),
      q1 = quantile(value, 0.25),
      q3 = quantile(value, 0.75),
      .groups = "drop"
    ) %>%
    mutate(time = "Before mandates")
  
  # after mandates
  df_post_long <- df2 %>%
    pivot_longer(everything()) %>%
    group_by(name) %>%
    summarise(
      median_importance = median(value),
      q1 = quantile(value, 0.25),
      q3 = quantile(value, 0.75),
      .groups = "drop"
    ) %>%
    mutate(time = "After mandates")
  
  # 合并 before / after
  df <- bind_rows(df_pre_long, df_post_long) %>%
    filter(!str_detect(name, "state")) %>%
    group_by(time) %>%
    arrange(desc(median_importance), .by_group = TRUE) %>%
    slice(1:10) %>%
    ungroup() %>%
    mutate(
      time = factor(time, levels = c("Before mandates", "After mandates"))
    )
  
  # before mandates 取负数，做 tornado plot
  df <- df %>%
    mutate(
      plot_value = ifelse(time == "Before mandates", -median_importance, median_importance),
      plot_q1 = ifelse(time == "Before mandates", -q1, q1),
      plot_q3 = ifelse(time == "Before mandates", -q3, q3)
    )
  
  # 给变量分类，尽量贴近 Ryan
  grouping_list <- c(
    "Self protective behaviours",
    "Demographics",
    "Health, mental health and wellbeing",
    "Perception of illness threat",
    "Time",
    "Trust in government"
  )
  
  df <- df %>%
    mutate(
      group_var = case_when(
        str_detect(name, "i2|i9|i11|protective") ~ grouping_list[1],
        str_detect(name, "age|house|employ|gender|state") ~ grouping_list[2],
        str_detect(name, "PHQ|cantril|d1") ~ grouping_list[3],
        str_detect(name, "r1") ~ grouping_list[4],
        str_detect(name, "week") ~ grouping_list[5],
        str_detect(name, "WCR") ~ grouping_list[6],
        TRUE ~ "Other"
      )
    )
  
  return(df)
}

# 标签美化，尽量贴近 Ryan
labels_clean <- function(labels_original){
  labels <- labels_original %>%
    str_replace_all("_", " ") %>%
    str_to_title()
  
  labels <- str_replace(labels, " Nomask Scale", "")
  
  labels[str_detect(labels, "State")] <- str_c(labels[str_detect(labels, "State")], ")")
  labels[str_detect(labels, "State")] <- str_replace(
    labels[str_detect(labels, "State")],
    "State ",
    "State ("
  )
  
  labels[str_detect(labels, "Gender")] <- str_c(labels[str_detect(labels, "Gender")], ")")
  labels[str_detect(labels, "Gender")] <- str_replace(
    labels[str_detect(labels, "Gender")],
    "Gender ",
    "Gender ("
  )
  
  labels[str_detect(labels, "I11 Health")] <- str_c(
    labels[str_detect(labels, "I11 Health")],
    " To Isolate"
  )
  labels[str_detect(labels, "I11 Health")] <- str_replace(
    labels[str_detect(labels, "I11 Health")],
    "I11 Health ",
    ""
  )
  
  labels[str_detect(labels, "Employment")] <- str_c(
    labels[str_detect(labels, "Employment")],
    ")"
  )
  labels[str_detect(labels, "Employment")] <- str_replace(
    labels[str_detect(labels, "Employment")],
    "Employment Status ",
    "Employment Status\n("
  )
  
  labels[str_detect(labels, "I9 Health")] <- str_c(
    labels[str_detect(labels, "I9 Health")],
    ")"
  )
  labels[str_detect(labels, "I9 Health")] <- str_replace(
    labels[str_detect(labels, "I9 Health")],
    "I9 Health ",
    "Isolate If Unwell ("
  )
  
  labels[str_detect(labels, "I2 Health")] <- "Non-Household Contacts"
  labels[str_detect(labels, "D1")] <- "Has Comorbidities"
  
  labels[str_detect(labels, "R1 1")] <- "Perceived Severity"
  labels[str_detect(labels, "R1 2")] <- "Perceived Susceptibility"
  
  labels[str_detect(labels, "Phq4")] <- str_c(labels[str_detect(labels, "Phq4")], ")")
  labels[str_detect(labels, "Phq4")] <- str_replace(
    labels[str_detect(labels, "Phq4")],
    "Phq4 1 ",
    "Little interest or pleasure\n("
  )
  labels[str_detect(labels, "Phq4")] <- str_replace(
    labels[str_detect(labels, "Phq4")],
    "Phq4 2 ",
    "Feeling down or depressed\n("
  )
  labels[str_detect(labels, "Phq4")] <- str_replace(
    labels[str_detect(labels, "Phq4")],
    "Phq4 3 ",
    "Feeling nervous or anxious\n("
  )
  labels[str_detect(labels, "Phq4")] <- str_replace(
    labels[str_detect(labels, "Phq4")],
    "Phq4 4 ",
    "Worrying ("
  )
  
  labels[str_detect(labels, "Wcrex2")] <- str_c(labels[str_detect(labels, "Wcrex2")], ")")
  labels[str_detect(labels, "Wcrex2")] <- str_replace(
    labels[str_detect(labels, "Wcrex2")],
    "Wcrex2 ",
    "Confidence in response\n("
  )
  
  return(labels)
}

# 配色：如果你不装 harrypotter，就直接用 brewer
get_colour_palette <- function(){
  groupings <- c(
    "Self protective behaviours",
    "Demographics",
    "Health, mental health and wellbeing",
    "Perception of illness threat",
    "Time",
    "Trust in government",
    "Other"
  )
  
  cols <- brewer.pal(7, "Set2")
  names(cols) <- groupings
  return(cols)
}

create_tornado_plot <- function(model_type, model_number){
  
  df <- load_data(model_type = model_type, model_number = model_number)
  
  # 为了让标签顺序稳定，按绝对值排序
  df <- df %>%
    mutate(abs_value = abs(plot_value)) %>%
    arrange(abs_value) %>%
    mutate(name = factor(name, levels = unique(name)))
  
  # 标签清洗
  labels_original <- levels(df$name)
  labels <- labels_clean(labels_original)
  
  colour_palette <- get_colour_palette()
  
  p <- df %>%
    ggplot(aes(x = plot_value, y = name, fill = group_var)) +
    geom_col() +
    geom_errorbar(aes(xmin = plot_q1, xmax = plot_q3), width = 0.5) +
    facet_wrap(~time, scales = "free_x") +
    scale_x_continuous(
      expand = c(0, 0),
      labels = function(x) signif(abs(x), 3)
    ) +
    scale_y_discrete(labels = labels) +
    scale_fill_manual(values = colour_palette) +
    guides(fill = guide_legend(nrow = 2)) +
    labs(
      y = NULL,
      x = "Median feature importance",
      fill = NULL
    ) +
    theme_bw() +
    theme(
      panel.spacing.x = unit(0, "mm"),
      legend.position = "none",
      text = element_text(size = text_size, family = "Times New Roman"),
      axis.text.y = element_text(size = 10),
      strip.text = element_text(size = 20),
      legend.text = element_text(size = 12),
      plot.margin = margin(t = 1, r = 15, b = 1, l = 1)
    )
  
  return(p)
}

# ===== Figure 1: Face mask wearing =====
model_number <- "model_1"

p_rf_1 <- create_tornado_plot(model_number = model_number, model_type = "rf")
p_xgb_1 <- create_tornado_plot(model_number = model_number, model_type = "xgboost") +
  theme(legend.position = "bottom")

p1 <- (p_rf_1 / p_xgb_1) +
  plot_annotation(tag_levels = "a", tag_suffix = ")", tag_prefix = "(")

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.png")),
  plot = p1,
  height = height,
  width = height,
  dpi = dpi
)

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.pdf")),
  plot = p1,
  height = height,
  width = height,
  dpi = dpi
)

# ===== Figure 2: General protective behaviour =====
model_number <- "model_2"

p_rf_2 <- create_tornado_plot(model_number = model_number, model_type = "rf")
p_xgb_2 <- create_tornado_plot(model_number = model_number, model_type = "xgboost") +
  theme(legend.position = "bottom")

p2 <- (p_rf_2 / p_xgb_2) +
  plot_annotation(tag_levels = "a", tag_suffix = ")", tag_prefix = "(")

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.png")),
  plot = p2,
  height = height,
  width = height,
  dpi = dpi
)

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.pdf")),
  plot = p2,
  height = height,
  width = height,
  dpi = dpi
)

# =========================================================================
# Figure 2: General protective behaviour
# =========================================================================

model_number <- "model_2"

p_rf_2 <- create_tornado_plot(
  model_number = model_number,
  model_type = "rf"
)

p_xgb_2 <- create_tornado_plot(
  model_number = model_number,
  model_type = "xgboost"
) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 13, colour = "black"),
    legend.key.size = unit(0.7, "cm"),
    legend.box.margin = margin(t = 5, b = 5)
  )

p2 <- (p_rf_2 / p_xgb_2) +
  plot_annotation(
    tag_levels = "a",
    tag_suffix = ")",
    tag_prefix = "(",
    theme = theme(
      plot.tag = element_text(
        size = 18,
        family = "Times New Roman",
        colour = "black"
      )
    )
  )

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.png")),
  plot = p2,
  height = height,
  width = width,
  dpi = dpi,
  bg = "white"
)

ggsave(
  here(glue("figures/mean_feature_importance_{model_number}_both_models.pdf")),
  plot = p2,
  height = height,
  width = width,
  dpi = dpi,
  bg = "white"
)
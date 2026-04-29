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
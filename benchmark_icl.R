cls_mtr <- metric_set(brier_class, roc_auc, mn_log_loss)

icl_uptake <- NULL

for (i in 1:75) {
  for (iter in 1:5) {
    tmp <- cls_tr |> slice_sample(n = i * 10)
    icl_fit <-
      brulee_tab_icl(
        class ~ A + B,
        data = tmp
      )
    icl_pred <- augment(icl_fit, cls_te)
    tmp_res <-
      icl_pred |>
      cls_mtr(class, .pred_red) |>
      mutate(size = i * 10)

    icl_uptake <- bind_rows(icl_uptake, tmp_res)
    rm(tmp, icl_fit, icl_pred, tmp_res)
  }
}

icl_uptake <- icl_uptake |>
  mutate(
    Metric = case_when(
      .metric == "brier_class" ~ "Brier Score",
      .metric == "roc_auc" ~ "ROC AUC",
      TRUE ~ "Cross-Entropy"
    )
  )
save(icl_uptake, file = "icl_uptake.RData")


icl_uptake |>
  ggplot(aes(size, .estimate)) +
  geom_point(alpha = 1 / 4, pch = 1) +
  facet_wrap(~ Metric, scales = "free_y") +
  geom_smooth(span = 0.2, se = FALSE, col = "red") +
  theme_bw()

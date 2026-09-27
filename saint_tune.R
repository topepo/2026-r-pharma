library(tidymodels)
library(tabby)
theme_set(theme_bw())

cores <- parallel::detectCores()

# ------------------------------------------------------------------------------

load("cls_data.RData")

set.seed(987)
cls_split <- initial_split(cls_data)
cls_split

cls_tr <- training(cls_split)
cls_te <- testing(cls_split)

set.seed(826)
cls_rs <- vfold_cv(cls_tr)

# ------------------------------------------------------------------------------

# Do a few tasks in parallel with the GPU but not too many (=too much GPU memory needed)
mirai::daemons(4)

# ------------------------------------------------------------------------------

saint_spec <-
  tabular_saint(
    num_embedding = tune(),
    hidden_units = tune(),
    num_attn_heads = tune(),
    num_attn_blocks = tune(),
    learn_rate = tune(),
    dropout_hidden = tune(),
    epochs = tune(),
    stop_iter = 10,
    batch_size = tune(),
    mode = "classification"
  ) |>
  set_engine("brulee", method = "SGD", device = "mps")

saint_wflow <- workflow(class ~ A + B, saint_spec)

# ------------------------------------------------------------------------------

system.time({
  set.seed(8272)
  saint_res <-
    saint_wflow |>
    tune_grid(
      resamples = cls_rs,
      grid = 20,
      metrics = metric_set(brier_class, roc_auc)
    )
})

autoplot(saint_res, metric = "brier_class")
select_best(saint_res, metric = "brier_class")


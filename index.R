## ------------------------------------------------------------------------------------------------
#| label: startup
#| include: false
library(brulee)
library(tidymodels)
theme_set(theme_bw())

pkg <- function(x, cran = TRUE) {
  cl <- match.call()
  x <- as.character(cl$x)
  pkg_chr(x, cran = cran)
}

pkg_chr <- function(x, cran = TRUE) {
  if (cran) {
    res <- glue::glue('<span class="pkg"><a href="https://cran.r-project.org/package={x}">{x}</a></span>')
  } else {
    res <- glue::glue('<span class="pkg">{x}</span>')
  }
 res 
}



## ------------------------------------------------------------------------------------------------
#| label: mlp-struc
#| echo: false
#| fig-align: center
#| out-width: "70%"

knitr::include_graphics("images/mlp.svg")


## ------------------------------------------------------------------------------------------------
#| label: mlp-start
#| echo: false
#| fig-align: center
#| out-width: "100%"

knitr::include_graphics("images/nnet_start.svg")


## ------------------------------------------------------------------------------------------------
#| label: act
#| echo: false
#| fig-align: center
#| fig-width: 8
#| fig-height: 2
#| out-width: "100%"

lp_grid <- seq(-3, 3, length.out = 100)

act_functions <-
  bind_rows(
    tibble(
      input = lp_grid,
      output = binomial()$linkinv(lp_grid),
      activation = "logistic"
    ),
    tibble(input = lp_grid, output = tanh(lp_grid), activation = "tanh"),
    tibble(
      input = lp_grid,
      output = as.numeric(torch::nnf_relu(lp_grid)),
      activation = "relu"
    ),
    tibble(
      input = lp_grid,
      output = as.numeric(torch::nnf_elu(lp_grid)),
      activation = "elu"
    ),
    tibble(
      input = lp_grid,
      output = as.numeric(torch::nnf_gelu(lp_grid)),
      activation = "gelu"
    )
  )

act_functions |>
  ggplot(aes(input, output)) +
  geom_line() +
  facet_wrap(~activation, scale = "free_y", nrow = 1) +
  theme_bw()


## ------------------------------------------------------------------------------------------------
#| label: mlp-end
#| echo: false
#| fig-align: center
#| out-width: "100%"

knitr::include_graphics("images/nnet_end.svg")


## ------------------------------------------------------------------------------------------------
#| label: data-splitting
#| code-line-numbers: "|1|2|4-6|10|11|"
library(tidymodels)
load("cls_data.RData")

set.seed(987)
cls_split <- initial_split(cls_data)
cls_split

cls_tr <- training(cls_split)
cls_te <- testing(cls_split)


## ------------------------------------------------------------------------------------------------
#| label: 2d-scat-code
#| eval: false
# p <-
#   cls_tr |>
#   ggplot(aes(A, B)) +
#   geom_point(
#     aes(col = class),
#     cex = 2, alpha = 3 / 4
#   ) +
#   coord_fixed() +
#   scale_color_brewer(palette = "Set1") +
#   theme(legend.position = "top")
# p


## ------------------------------------------------------------------------------------------------
#| label: 2d-scat
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
p <-
  cls_tr |>
  ggplot(aes(A, B)) +
  geom_point(
    aes(col = class), 
    cex = 2, alpha = 2 / 3
  ) +
  coord_fixed() +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "top")
p


## ------------------------------------------------------------------------------------------------
#| label: init-mlp
#| code-line-numbers: "|1|2|4-5|6|7|8-9|10-11|"
library(brulee)
set.seed(7826)
init_mlp <- brulee_mlp(
  class ~ A + B,          # Can also use a recipe or x/y format for data
  data = cls_tr,
  hidden_units = 6,
  learn_rate = 0.1,
  epochs = 100,
  stop_iter = 5,          # Default: randomly take 10% as internal validation
  batch_size = 32,
  method = "SGD"
)


## ------------------------------------------------------------------------------------------------
#| label: mlp-loss
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"

autoplot(init_mlp)


## ------------------------------------------------------------------------------------------------
#| label: mlp-details
summary(init_mlp)


## ------------------------------------------------------------------------------------------------
#| label: init-pred

# prediction: 
grid_mlp <- augment(init_mlp, new_data = cls_grid) 

grid_mlp |> slice_head(n = 5)


## ------------------------------------------------------------------------------------------------
#| label: mlp-plot-code
#| eval: false
# p +
#   geom_contour(
#     data = grid_mlp,
#     aes(z = .pred_red),
#     breaks = 1 / 2,
#     col = "black",
#     linewidth = 1
#   )


## ------------------------------------------------------------------------------------------------
#| label: mlp-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
p + 
  geom_contour(
    data = grid_mlp,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )


## ------------------------------------------------------------------------------------------------
#| label: three-layer-mlp
#| code-line-numbers: "|6-7|"
library(brulee)
set.seed(7826)
three_mlp <- brulee_mlp(
  class ~ A + B,          
  data = cls_tr,
  hidden_units = c(6, 2, 6),
  activation = c("relu", "relu", "tanh"),
  learn_rate = 0.1,
  epochs = 100,
  stop_iter = 5,          
  batch_size = 32,
  method = "SGD"
)


## ------------------------------------------------------------------------------------------------
#| label: three-loss
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"

autoplot(three_mlp)


## ------------------------------------------------------------------------------------------------
#| label: three-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
grid_three <- augment(three_mlp, new_data = cls_grid) 

p + 
  geom_contour(
    data = grid_three,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  ) + 
  geom_contour(
    data = grid_mlp,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    alpha = 1 / 3,
    linewidth = 1
  )


## ------------------------------------------------------------------------------------------------
#| label: ml-tune
#| code-line-numbers: "|1-2|4-6|8|13|15|"
#| eval: false
# set.seed(826)                # Create resamples of the training set
# cls_rs <- vfold_cv(cls_tr)
# 
# mlp_spec <-                  # Optimize some parameters
#   mlp(mode = "classification", hidden_units = tune(), epochs = 100) |>
#   set_engine("brulee")
# 
# mirai::daemons(10)           # For parallel processing
# set.seed(658)
# mlp_res <-
#   mlp_spec |>
#   tune_grid(
#     class ~ A + B,           # Could be x/y, or a recipe.
#     resamples = cls_rs,
#     grid = 25                # Specify a grid or how many candidates to test
#   )


## ------------------------------------------------------------------------------------------------
#| label: resnet-fit
#| cache: true
#| code-line-numbers: "|7|13|"
set.seed(7826)
resnet_fit <- brulee_resnet(
  class ~ A + B,             
  data = cls_tr,
  hidden_units = c(6, 2, 6), 
  activation = c("relu", "relu", "tanh"),
  residual_at = 1:2,         # residual after these layers
  learn_rate = 0.01,
  epochs = 100,
  stop_iter = 5,  
  batch_size = 32,
  method = "SGD",
  device = "cpu"             # default; results are different w/gpu, see below
)


## ------------------------------------------------------------------------------------------------
#| label: resnet-loss
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"

autoplot(resnet_fit)


## ------------------------------------------------------------------------------------------------
#| label: resnet-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
grid_resnet <- 
  augment(resnet_fit, new_data = cls_grid) |> 
  select(A, B, .pred_class, .pred_red)

p + 
  geom_contour(
    data = grid_resnet,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )


## ------------------------------------------------------------------------------------------------
#| label: devices
#| echo: false
#| fig-align: center
#| fig-width: 7
#| fig-height: 5
#| out-width: "100%"
load("losses.RData")

means <- 
  losses |> 
  slice_head(n = 1, by = c(iter)) |> 
  summarize(mean = mean(time_per_epoch), .by = c(Device)) |> 
  arrange(Device)

losses |> 
  mutate(
    iter = format(iter),
    Device = ifelse(Device == "MPS", "GPU", Device)
  ) |> 
  ggplot(aes(Epoch, Loss, col = Device, pch = Device, group = iter)) + 
  geom_line(alpha = 1 / 2) + 
  geom_point() +
  theme(legend.position = "top")


## ------------------------------------------------------------------------------------------------
#| label: saint-fit
#| cache: true
#| code-line-numbers: "|6-11|16"
set.seed(7826)
saint_fit <- 
  brulee_saint(
    class ~ A + B,
    data = cls_tr, 
    num_embedding = 20,    # Ok, I admit that I used the tabby and tune packages
    hidden_units = 20,     # to find this combination of parameters. It took a
    num_attn_heads = 8,    # long time to run. 
    num_attn_blocks = 4,
    learn_rate = 0.005,    # learn_rate and dropout for hidden units were 
    dropout_hidden = 0,    # unusually low for these data. 
    epochs = 100,
    stop_iter = 10,  
    batch_size = 16,
    method = "SGD",
    device = "mps"
  )


## ------------------------------------------------------------------------------------------------
#| label: saint-loss
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"

autoplot(saint_fit)


## ------------------------------------------------------------------------------------------------
#| label: saint-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
#| cache: true
grid_saint <- 
  augment(saint_fit, new_data = cls_grid) |> 
  select(A, B, .pred_class, .pred_red)

p + 
  geom_contour(
    data = grid_saint,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )


## ------------------------------------------------------------------------------------------------
#| label: saint-summary
#| cache: true
nrow(cls_tr)
summary(saint_fit)


## ------------------------------------------------------------------------------------------------
#| label: scm
#| echo: false
#| fig-align: "center"
knitr::include_graphics("images/dag_combined.png")


## ------------------------------------------------------------------------------------------------
#| label: tabicl
#| results: hide

icl_fit <- 
  brulee_tab_icl(
    class ~ A + B, 
    data = cls_tr
  ) # That's it!


## ------------------------------------------------------------------------------------------------
#| label: tabicl-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "100%"
#| cache: true
grid_tabicl <- 
  augment(icl_fit, new_data = cls_grid) |> 
  select(A, B, .pred_class, .pred_red)

p + 
  geom_contour(
    data = grid_tabicl,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )


## ------------------------------------------------------------------------------------------------
#| label: icl
#| echo: false
#| fig-align: center
#| fig-width: 7
#| fig-height: 3.5
#| out-width: "40%"
load("icl_uptake.RData")

icl_uptake |>
  ggplot(aes(size, .estimate)) +
  geom_point(alpha = 1 / 4, pch = 1) +
  facet_wrap(~ Metric, scales = "free_y") +
  geom_smooth(span = 0.2, se = FALSE, col = "red") +
  theme_bw() +
  labs(x = "Training Set Size", y = "Out-of-Sample Performance")


## ------------------------------------------------------------------------------------------------
#| label: noise
#| echo: false
#| fig-align: center
#| out-width: "70%"

knitr::include_graphics("images/irrelevant-predictors.svg")


## ------------------------------------------------------------------------------------------------
#| label: tabicl-test
#| cache: false
cls_mtr <- metric_set(brier_class, roc_auc, mn_log_loss)

icl_test_pred <- augment(icl_fit, cls_te)

icl_test_pred |> cls_mtr(class, .pred_red)


## ------------------------------------------------------------------------------------------------
#| label: tabicl-test-plot
#| echo: false
#| fig-align: center
#| fig-width: 5
#| fig-height: 5
#| out-width: "50%"
#| cache: true
cls_te |>
  ggplot(aes(A, B)) +
  geom_point(
    aes(col = class), 
    cex = 2, alpha = 3 / 4
  ) +
  coord_fixed() +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "top") + 
  geom_contour(
    data = grid_tabicl,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )


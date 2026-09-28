## ------------------------------------------------------------------------------------------------

library(brulee)
library(tidymodels)
theme_set(theme_bw())

## ------------------------------------------------------------------------------------------------

load("cls_data.RData")

set.seed(987)
cls_split <- initial_split(cls_data)
cls_split

cls_tr <- training(cls_split)
cls_te <- testing(cls_split)

## ------------------------------------------------------------------------------------------------

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

autoplot(init_mlp)

summary(init_mlp)

## ------------------------------------------------------------------------------------------------

grid_mlp <- augment(init_mlp, new_data = cls_grid) 

grid_mlp |> slice_head(n = 5)

## ------------------------------------------------------------------------------------------------

p + 
  geom_contour(
    data = grid_mlp,
    aes(z = .pred_red),
    breaks = 1 / 2,
    col = "black",
    linewidth = 1
  )

## ------------------------------------------------------------------------------------------------

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

autoplot(three_mlp)

## ------------------------------------------------------------------------------------------------

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

autoplot(resnet_fit)

## ------------------------------------------------------------------------------------------------

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

autoplot(saint_fit)
nrow(cls_tr)
summary(saint_fit)

## ------------------------------------------------------------------------------------------------

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

icl_fit <- 
  brulee_tab_icl(
    class ~ A + B, 
    data = cls_tr
  ) # That's it!

## ------------------------------------------------------------------------------------------------

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

cls_mtr <- metric_set(brier_class, roc_auc, mn_log_loss)

icl_test_pred <- augment(icl_fit, cls_te)

icl_test_pred |> cls_mtr(class, .pred_red)

## ------------------------------------------------------------------------------------------------

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

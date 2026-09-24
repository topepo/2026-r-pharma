system.time({
  set.seed(7826)
  torch::torch_manual_seed(7826)
  cpu1 <- resnet_fit <- brulee_resnet(
    class ~ A + B,
    data = cls_tr,
    hidden_units = c(6, 2, 6),
    residual_at = 1:2,
    learn_rate = 0.01,
    epochs = 100,
    stop_iter = 5,
    batch_size = 32,
    method = "SGD"
  )
})

system.time({
  set.seed(7826)
  torch::torch_manual_seed(7826)
  cpu2 <- resnet_mps_fit <- brulee_resnet(
    class ~ A + B,
    data = cls_tr,
    hidden_units = c(6, 2, 6),
    residual_at = 1:2,
    learn_rate = 0.01,
    epochs = 100,
    stop_iter = 5,
    batch_size = 32,
    method = "SGD",
    device = "mps"
  )

})

cpu_loss <- resnet_fit$loss
mps_loss <- resnet_mps_fit$loss

losses <-
  tibble(
    Loss = c(cpu_loss, mps_loss),
    Epoch = c(seq_along(cpu_loss) - 1, seq_along(mps_loss) - 1),
    Device = rep(c("CPU", "MPS"), c(length(cpu_loss), length(mps_loss)))
  )
losses |>
  ggplot(aes(Epoch, Loss, col = Device, pch = Device)) +
  geom_line() +
  geom_point()

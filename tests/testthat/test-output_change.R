
test_that("testing if gibbs_without_u_screen result changed", {
  expect_snapshot_value(c(dpr_model_gibbs$alpha, dpr_model_gibbs$beta), style = "serialize")
})

test_that("testing if VB result changed", {
  expect_snapshot_value(c(dpr_model_vb$alpha, dpr_model_vb$beta), style = "serialize")
})

test_that("testing if adaptive_gibbs_without_u_screen result changed", {
  expect_snapshot_value(c(dpr_model_adaptive_gibbs$alpha, dpr_model_adaptive_gibbs$beta), style = "serialize")
})

test_that("testing if VB without kinship result changed", {
  expect_snapshot_value(c(dpr_model_vb_no_kinship$alpha, dpr_model_vb_no_kinship$beta), style = "serialize")
})

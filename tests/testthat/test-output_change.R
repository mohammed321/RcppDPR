
test_that("testing if gibbs_without_u_screen result changed", {
  expect_snapshot_value(dpr_model_gibbs, style = "serialize")
})

test_that("testing if VB result changed", {
  expect_snapshot_value(dpr_model_vb, style = "serialize")
})

test_that("testing if adaptive_gibbs_without_u_screen result changed", {
  expect_snapshot_value(dpr_model_adaptive_gibbs, style = "serialize")
})

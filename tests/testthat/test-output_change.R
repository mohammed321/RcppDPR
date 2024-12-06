
test_that("testing if gibbs_without_u_screen result changed", {
  expect_snapshot_value(gibbs_without_u_screen_result, style = "serialize")
})

test_that("testing if VB result changed", {
  expect_snapshot_value(VB_result, style = "serialize")
})

test_that("testing if adaptive_gibbs_without_u_screen result changed", {
  expect_snapshot_value(adaptive_gibbs_without_u_screen_result, style = "serialize")
})

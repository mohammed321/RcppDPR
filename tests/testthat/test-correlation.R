test_that("testing if methods produce correlated results alpha", {
  testthat::expect_gt(cor(gibbs_without_u_screen_result$alpha, VB_result$alpha), 0.9)

  testthat::expect_gt(cor(gibbs_without_u_screen_result$alpha, adaptive_gibbs_without_u_screen_result$alpha), 0.9)

  testthat::expect_gt(cor(VB_result$alpha, adaptive_gibbs_without_u_screen_result$alpha), 0.9)
})

test_that("testing if methods produce correlated results beta", {
  testthat::expect_gt(cor(gibbs_without_u_screen_result$beta, VB_result$beta), 0.9)

  testthat::expect_gt(cor(gibbs_without_u_screen_result$beta, adaptive_gibbs_without_u_screen_result$beta), 0.9)

  testthat::expect_gt(cor(VB_result$beta, adaptive_gibbs_without_u_screen_result$beta), 0.9)
})


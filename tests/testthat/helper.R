get_results <- function() {
  file_path_x <- system.file("extdata", "data/in/x.rds", package = "RcppDPR")
  print(file_path_x)
  x = readRDS(file_path_x)
  file_path_y <- system.file("extdata", "data/in/y.rds", package = "RcppDPR")
  y = readRDS(file_path_y)
  file_path_w <- system.file("extdata", "data/in/w.rds", package = "RcppDPR")
  w = readRDS(file_path_w)

  set.seed(42)
  dpr_model_gibbs <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "Gibbs", show_progress = FALSE)

  set.seed(42)
  dpr_model_vb <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "VB", show_progress = FALSE)

  set.seed(42)
  dpr_model_adaptive_gibbs <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "Adaptive_Gibbs", show_progress = FALSE)

  set.seed(42)
  dpr_model_vb_no_kinship <<- fit_model(y,w,x, show_progress = FALSE)
}

get_results()

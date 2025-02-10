get_results <- function() {

  x = readRDS("data/in/x.rds")
  y = readRDS("data/in/y.rds")
  w = readRDS("data/in/w.rds")

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

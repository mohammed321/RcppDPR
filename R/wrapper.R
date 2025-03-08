#' Fit Dirichlet Process Regression model
#'
#' @param y Numeric vector of outcome
#' @param x Numeric matrix of predictors
#' @param w Numeric matrix of covariates (default = rep(1, length(y)))
#' @param rotate_variables Logical value indicating whether to rotate y, w and x using covariance_matrix (default = FALSE)
#' @param covariance_matrix Numeric sample covariance matrix used for rotation of y, w and x - if NULL and rotate_variables is TRUE then the sample covariance matrix is computed from x
#' @param fitting_method Character string indicating the method used for fitting the data - possible values are:
#' * 'Gibbs' - full Bayesian inference with Gibbs sampler with a fixed n_k
#' * 'Adaptive_Gibbs' - adaptive version of Gibbs sample that automatically chooses n_k
#' * 'VB' - variational Bayes inference with a fixed n_k
#' @param ... arguments to pass through to internal methods.
#'
#' @return returns an object of class 'DPR_Model'
#'
#' @useDynLib RcppDPR, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @export
#'
#' @description
#' Fit a Dirichlet Process Regression model using a specified fitting method.  Outcome (y) should be Gaussian and scaled and centered; predictors (x) and covariates (w) should also be scaled and centered but may be of any distribution
#'
#' @details
#' fit_model() can pass a number of additional parameters to the different fitting methods. These parameters are used for all modes:
#' * n_k: number of mixture components in scale mixture of normals prior (default = 4)
#' * l_min: minimum value of log-likelihood for initial parameter search (default = 1e-7, only modify if you know what you are doing)
#' * l_max: maximum value of log-likelihood for initial parameter search (default = 1e5, only modify if you know what you are doing)
#' * n_regions: number of regions over which to search for maximum log-likelihood (default = 10, only modify if you know what you are doing)
#'
#' These parameters are only used for the Gibbs and Adaptive Gibbs modes:
#' * w_step: number of burn-in steps for Gibbs sampler (default = 1000)
#' * s_step: number of inference steps for Gibbs sampler (default = 1000)
#' * m_n_k: maximum number of mixture components in scale mixture of normals prior (default = 6, Adaptive Gibbs only)
#' @examples
#' file_path_x <- system.file("extdata", "data/in/x.rds", package = "RcppDPR")
#' file_path_y <- system.file("extdata", "data/in/y.rds", package = "RcppDPR")
#' file_path_w <- system.file("extdata", "data/in/w.rds", package = "RcppDPR")
#' x = readRDS(file_path_x)
#' y = readRDS(file_path_y)
#' w = readRDS(file_path_w)
#' dpr_model <- fit_model(y, w, x, fitting_method = "VB")
fit_model <- function(y, w, x, rotate_variables = FALSE, covariance_matrix = NULL, fitting_method = "VB", ...) {

  if (rotate_variables == FALSE) {

    if (fitting_method == "VB") {

      model <- run_VB_no_kinship(y, w, x, ...)
    } else if (fitting_method == "Gibbs") {

      model <- run_gibbs_without_u_screen_no_kinship(y, w, x, ...)
    } else if (fitting_method == "Adaptive_Gibbs") {

      model <- run_gibbs_without_u_screen_adaptive_no_kinship(y, w, x, ...)
    } else {

      stop("Invalid fitting method. Choose 'VB', 'Gibbs', or 'Adaptive_Gibbs'.")
    }
  } else {

    if (is.null(covariance_matrix)) {

      if (fitting_method == "VB") {

        model <- run_VB(y, w, x, ...)
      } else if (fitting_method == "Gibbs") {

        model <- run_gibbs_without_u_screen(y, w, x, ...)
      } else if (fitting_method == "Adaptive_Gibbs") {

        model <- run_gibbs_without_u_screen_adaptive(y, w, x, ...)
      } else {

        stop("Invalid fitting method. Choose 'VB', 'Gibbs', or 'Adaptive_Gibbs'.")
      }
    } else {

      if (fitting_method == "VB") {

        model <- run_VB_custom_kinship(y, w, x, covariance_matrix, ...)
      } else if (fitting_method == "Gibbs") {

        model <- run_gibbs_without_u_screen_custom_kinship(y, w, x, covariance_matrix, ...)
      } else if (fitting_method == "Adaptive_Gibbs") {

        model <- run_gibbs_without_u_screen_adaptive_custom_kinship(y, w, x, covariance_matrix, ...)
      } else {

        stop("Invalid fitting method. Choose 'VB', 'Gibbs', or 'Adaptive_Gibbs'.")
      }
    }


  }

    model$y <- y
    model$x <- x
    model$w <- w

    class(model) <- "DPR_Model"

    return(model)
}

#' Use a DPR model to predict results from new data
#'
#' @param object an object of class DPR_Model
#' @param newdata Numeric matrix representing the input to the model
#' @param ... ignored args.
#'
#' @return returns Numeric vector of predictions
#'
#' @export
#' @examples
#' n <- 500
#' p <- 10775
#' file_path_x <- system.file("extdata", "data/in/x.rds", package = "RcppDPR")
#' file_path_y <- system.file("extdata", "data/in/y.rds", package = "RcppDPR")
#' file_path_w <- system.file("extdata", "data/in/w.rds", package = "RcppDPR")
#' x = readRDS(file_path_x)
#' y = readRDS(file_path_y)
#' w = readRDS(file_path_w)
#' dpr_model <- fit_model(y, w, x, fitting_method = "VB")
#' new_x <- matrix(rnorm(n = n * p, mean = 0, sd = 1), nrow = n, ncol = p)
#' new_y <- predict(dpr_model, new_x)
predict.DPR_Model <- function(object, newdata, ...) {
  if (missing(newdata)) {
    stop("newdata must be provided for prediction.")
  }

  return((newdata %*% (object$beta + object$alpha)) + object$pheno_mean)
}

#' fit a model to the data using a specified fitting method
#'
#' @param y Numeric vector
#' @param w Numeric matrix
#' @param x Numeric matrix
#' @param rotate_variables Logical value indicating whether to rotate y,w,x using covariance_matrix
#' @param covariance_matrix Numeric matrix used for rotation of y,w,x if NULL and rotate_variables is TRUE then default matrix is used
#' @param fitting_method Character string indicating the method used for fitting the data possible values are 'VB' 'Gibbs' 'Adaptive_Gibbs'
#' @param ... arguments to pass down to internal methods.
#'
#' @return returns an object of class 'DPR_Model'
#'
#' @useDynLib RcppDPR, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @export
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

#' use a DPR model to predict rsults from newdata
#'
#' @param dpr_model an object of class DPR_Model
#' @param newdata Numeric matrix representing the input to the model
#'
#' #' @return returns Numeric vector of predictions
#'
#' @export
predict.DPR_Model <- function(dpr_model, newdata) {
  if (missing(newdata)) {
    stop("newdata must be provided for prediction.")
  }

  return((newdata %*% (dpr_model$beta + dpr_model$alpha)) + dpr_model$pheno_mean)
}

runDPR <- function(y, w, x, model_name = "VB") {
    if (model_name == "VB") {
      return(run_VB(y, w, x))
    }
    else if (model_name == "Gibbs") {
      return(run_gibbs_without_u_screen(y, w, x))
    }
    else if (model_name == "Adaptive_Gibbs") {
      return(run_gibbs_without_u_screen_adaptive(y, w, x))
    }
    else {
      stop("Invalid model name. Choose 'VB', 'Gibbs', or 'Adaptive_Gibbs'.")
    }
}

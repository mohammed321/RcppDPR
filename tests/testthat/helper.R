library("snpStats")

get_results <- function() {
  plink_data <- read.plink("data/in/mouse_hs1940.bed",
                           "data/in/mouse_hs1940.bim",
                           "data/in/mouse_hs1940.fam")

  mask <- !is.na(as.numeric(plink_data$fam[,7]))
  y <- as.numeric(plink_data$fam[mask,7])

  x <- plink_data$genotypes[mask,]
  x <- matrix(as.numeric(x), nrow(x), ncol(x))
  x_0 <- (x == 0)
  x_1 <- (x == 1)
  x_2 <- (x == 2)
  x_3 <- (x == 3)
  x[x_0] <- NA #missing val
  x[x_1] <- 2.0
  x[x_2] <- 1.0
  x[x_3] <- 0

  # Identify columns to keep
  cols_to_keep <- apply(x, 2, function(col) {
    mean(is.na(col)) <= 0.05
  })
  x <- x[,cols_to_keep]

  cols_to_keep <- apply(x, 2, function(col) {
    val = mean(col, na.rm = TRUE)
    0.02 <= val && val <= 1.98
  })
  x <- x[,cols_to_keep]

  x <- apply(x, 2, function(col) {col - mean(col, na.rm = TRUE)})
  x[is.na(x)] <- 0

  w <- matrix(1.0, nrow = length(y), ncol = 1)

  set.seed(42)
  dpr_model_gibbs <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "Gibbs", show_progress = FALSE)

  set.seed(42)
  dpr_model_vb <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "VB", show_progress = FALSE)

  set.seed(42)
  dpr_model_adaptive_gibbs <<- fit_model(y,w,x, rotate_variables = TRUE, fitting_method = "Adaptive_Gibbs", show_progress = FALSE)
}

get_results()

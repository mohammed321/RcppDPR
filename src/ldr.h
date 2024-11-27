#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using arma::mat;
using arma::vec;

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen(
    arma::vec &y,
    arma::mat &W,
    arma::mat &UtX,
    size_t n_k = 4,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10
    );

// [[Rcpp::export]]
Rcpp::List run_VB(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k = 4,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10
    );
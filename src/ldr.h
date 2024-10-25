#include <armadillo>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

using arma::ivec;
using arma::mat;
using arma::vec;

// [[Rcpp::export]]
void run(
    vec &y,
    mat &W,
    mat &G, // ZP
    mat &UtX,
    size_t ni_test,
    size_t n_cvt,
    size_t ns_test,
    double l_min,
    double l_max,
    size_t n_region,
    double l_remle_null,
    double logl_remle_H0,
    double pve_null,
    double pve_se_null,
    double vg_remle_null,
    double ve_remle_null,
    size_t n_k,
    size_t w_step,
    size_t s_step
    );
#include <RcppArmadillo.h>

using arma::mat;
using arma::vec;

// [[Rcpp::export]]
Rcpp::List run_VB(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k = 4,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_VB_custom_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    arma::mat &G,
    size_t n_k = 4,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_VB_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k = 4,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k = 4,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen_custom_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    arma::mat &G,
    size_t n_k = 4,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k = 4,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen_adaptive(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t m_n_k = 6,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen_adaptive_custom_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    arma::mat &G,
    size_t m_n_k = 6,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );

// [[Rcpp::export]]
Rcpp::List run_gibbs_without_u_screen_adaptive_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t m_n_k = 6,
    size_t w_step = 1000,
    size_t s_step = 1000,
    double l_min = 1e-7,
    double l_max = 1e5,
    size_t n_region = 10,
    bool show_progress = true
    );
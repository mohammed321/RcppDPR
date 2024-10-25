#include <armadillo>
#include <gsl/gsl_matrix.h>
#include <memory>

std::unique_ptr<gsl_matrix> arma_mat_to_gsl_mat(const arma::mat& arma_mat);
arma::mat gsl_mat_to_arma_mat(const gsl_matrix* gsl_mat);

std::unique_ptr<gsl_vector> arma_vec_to_gsl_vec(const arma::vec& arma_vec);
arma::vec gsl_vec_to_arma_vec(const gsl_vector* gsl_vec);
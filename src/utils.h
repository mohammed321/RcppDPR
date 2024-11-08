#include <iostream>
#include <RcppArmadillo.h>
#include <gsl/gsl_matrix.h>
#include <memory>
#include <string>
#include <fstream>

std::unique_ptr<gsl_matrix> arma_mat_to_gsl_mat(const arma::mat& arma_mat);
arma::mat gsl_mat_to_arma_mat(const gsl_matrix* gsl_mat);

std::unique_ptr<gsl_vector> arma_vec_to_gsl_vec(const arma::vec& arma_vec);
arma::vec gsl_vec_to_arma_vec(const gsl_vector* gsl_vec);

class ProgressBar {
    size_t m_total;
    size_t m_current;
    size_t m_num_of_bars;
    const std::string m_label;

public:
    ProgressBar(const std::string label, size_t total, size_t num_of_bars = 50);
    void advance();
    friend std::ostream& operator<<(std::ostream& out, const ProgressBar& pb);
};

std::ofstream get_output_file(const char* file_name = "ouput.txt");
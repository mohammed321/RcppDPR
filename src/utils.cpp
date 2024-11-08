#include "utils.h"
#include <filesystem>

std::unique_ptr<gsl_matrix> arma_mat_to_gsl_mat(const arma::mat& arma_mat)
{
    // Get the dimensions of the Armadillo matrix
    size_t rows = arma_mat.n_rows;
    size_t cols = arma_mat.n_cols;

    // Create a GSL matrix with the same dimensions
    gsl_matrix* gsl_mat = gsl_matrix_alloc(rows, cols);

    // Copy the elements from Armadillo matrix to GSL matrix
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            gsl_matrix_set(gsl_mat, i, j, arma_mat.at(i, j));
        }
    }

    return std::unique_ptr<gsl_matrix>(gsl_mat);
}

std::unique_ptr<gsl_vector> arma_vec_to_gsl_vec(const arma::vec& arma_vec)
{
    // Get the dimensions of the Armadillo vector
    size_t n_elem = arma_vec.n_elem;

    // Create a GSL vector with the same dimensions
    gsl_vector* gsl_vec = gsl_vector_alloc(n_elem);

    // Copy the elements from Armadillo vector to GSL matrix
    for (size_t i = 0; i < n_elem; ++i) {
        gsl_vector_set(gsl_vec, i, arma_vec.at(i));
    }

    return std::unique_ptr<gsl_vector>(gsl_vec);
}

// Function to convert gsl_matrix to arma::mat
arma::mat gsl_mat_to_arma_mat(const gsl_matrix* gsl_mat)
{
    size_t rows = gsl_mat->size1;  // Number of rows in the GSL matrix
    size_t cols = gsl_mat->size2;  // Number of columns in the GSL matrix

    // Create an Armadillo matrix with the same dimensions
    arma::mat arma_mat(rows, cols);

    // Copy elements from GSL matrix to Armadillo matrix
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            arma_mat.at(i, j) = gsl_matrix_get(gsl_mat, i, j);
        }
    }

    // Return the converted Armadillo matrix
    return arma_mat;
}

// Function to convert gsl_vector to arma::vec
arma::vec gsl_vec_to_arma_vec(const gsl_vector* gsl_vec)
{
    size_t n_elem = gsl_vec->size;  // Number of elements in the GSL vector

    // Create an Armadillo vector with the same dimensions
    arma::vec arma_vec(n_elem);

    // Copy elements from GSL vector to Armadillo vector
    for (size_t i = 0; i < n_elem; ++i) {
        arma_vec.at(i) = gsl_vector_get(gsl_vec, i);
    }

    // Return the converted Armadillo matrix
    return arma_vec;
}

ProgressBar::ProgressBar(const std::string label, size_t total, size_t num_of_bars) : 
m_total(total), m_current(0), m_num_of_bars(num_of_bars), m_label(label) {}

void ProgressBar::advance()
{
    m_current++;
}

std::ostream& operator<<(std::ostream& out, const ProgressBar& pb)
{
    out << '\r' << pb.m_label << " [";
    for (size_t i = 0; i < pb.m_num_of_bars; i++) {
        out << ((i < ((pb.m_num_of_bars*pb.m_current)/pb.m_total))? '#' : ' '); 
    }
    out << "] " << std::setprecision(2) << std::setw(6) << std::fixed << 100.0 * static_cast<double>(pb.m_current)/pb.m_total << '%';
    return out;
}

std::ofstream get_output_file(const char* file_name) {
    namespace fs = std::filesystem;

    // Convert the full path string to a filesystem path
    fs::path file_path(file_name);

    // Extract the folder path (parent path)
    fs::path folder_path = file_path.parent_path();

    // Create the folder if it doesn't already exist
    if (!folder_path.empty() && !fs::exists(folder_path)) {
        fs::create_directories(folder_path);
        std::cout << "Folder created: " << folder_path << std::endl;
    } else {
        std::cout << "Folder already exists or no folder specified." << std::endl;
    }

    std::ofstream outfile(file_name);
    if (!outfile.is_open()) {
        std::cerr << "Could not open file for writing." << std::endl;
        exit(1);
    } 

    return outfile;
}

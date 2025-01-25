#include "utils.h"
#include <filesystem>


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
        Rcpp::Rcout << "Folder created: " << folder_path << std::endl;
    } else {
        Rcpp::Rcout << "Folder already exists or no folder specified." << std::endl;
    }

    std::ofstream outfile(file_name);
    if (!outfile.is_open()) {
        Rcpp::Rcerr << "Could not open file for writing." << std::endl;
        return std::ofstream();
    } 

    return outfile;
}

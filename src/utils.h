#include <iostream>
#include <RcppArmadillo.h>
#include <memory>
#include <string>
#include <fstream>


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
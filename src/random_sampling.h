#include <gsl/gsl_rng.h>

class RandomSampler {
    public:
    RandomSampler(const unsigned long int seed = 0);
    ~RandomSampler();

    double gamma_sample(const double a, const double b);
    double beta_sample(const double a, const double b);
    double gaussian_sample(const double sigma);
    void multinomial_sample(size_t K, unsigned int N, const double p[], unsigned int n[]);
    double uniform_sample();
    double beta_pdf_val(const double x, const double a, const double b);

    private:
    gsl_rng* m_rng;
};
#include "random_sampling.h"
#include <gsl/gsl_randist.h>

RandomSampler::RandomSampler(const unsigned long int seed) {
    m_rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(m_rng, seed);
};

RandomSampler::~RandomSampler() {
    gsl_rng_free(m_rng);
}

double RandomSampler::gamma_sample(const double a, const double b) {
    return gsl_ran_gamma(m_rng, a, b);
}

double RandomSampler::beta_sample(const double a, const double b) {
    return gsl_ran_beta(m_rng, a, b);
}

double RandomSampler::gaussian_sample(const double sigma) {
    return gsl_ran_gaussian(m_rng, sigma);
}

void RandomSampler::multinomial_sample(size_t K, unsigned int N, const double p[], unsigned int n[]) {
    gsl_ran_multinomial(m_rng, K, N, p, n);
}

double RandomSampler::beta_pdf_val(const double x, const double a, const double b) {
    return gsl_ran_beta_pdf(x, a, b);
}

double RandomSampler::uniform_sample() {
    return gsl_rng_uniform(m_rng);
}
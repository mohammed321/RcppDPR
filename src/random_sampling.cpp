#include "random_sampling.h"
#include <Rcpp.h>

double RandomSampler::gamma_sample(const double a, const double b) {
    return R::rgamma(a, b);
}

double RandomSampler::beta_sample(const double a, const double b) {
    return R::rbeta(a, b);
}

double RandomSampler::gaussian_sample(const double sigma) {
    return R::rnorm(0.0, sigma);
}

void RandomSampler::multinomial_sample(int K, int N, double p[], int n[]) {
    R::rmultinom(N, p, K, n);
}

double RandomSampler::beta_pdf_val(const double x, const double a, const double b) {
    return R::dbeta(x, a, b, false);
}

double RandomSampler::uniform_sample() {
    return R::runif(0.0, 1.0);
}

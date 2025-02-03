#include <RcppArmadillo.h>

using arma::mat;
using arma::vec;

namespace gibbs_without_u_screen_NS {

    struct Result {
        vec alpha;
        vec beta;
        vec post_Ealpha;
        double pheno_mean;
        double pD1;
        double pD2;
        double DIC1;
        double DIC2;
        double BIC1;
        double BIC2;
    };

    Result gibbs_without_u_screen(
        const mat &UtX,
        const vec &Uty,
        const mat &UtW,
        const vec &D,
        const vec &Wbeta,
        const vec &se_Wbeta,
        const vec &beta,
        double lambda,
        size_t n_k,
        size_t w_step,
        size_t s_step,
        bool show_progress);

    Result gibbs_without_u_screen_adaptive(
        const mat &UtX,
        const vec &Uty,
        const mat &UtW,
        const vec &eigen_values,
        const vec &Wbeta,
        const vec &se_Wbeta,
        const vec &beta,
        double lambda,
        size_t m_n_k,
        size_t w_step,
        size_t s_step,
        bool show_progress);
}

namespace VB_NS {

    struct Result {
        vec alpha;
        vec beta;
        double pheno_mean;
        vec ELBO;
    };

    Result VB(
        const mat &UtX,
        const vec &Uty,
        const mat &UtW,
        const vec &D,
        const vec &Wbeta,
        const vec &se_Wbeta,
        const vec &beta,
        double lambda,
        size_t n_k,
        bool show_progress);
}


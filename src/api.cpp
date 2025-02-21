#include "api.h"
#include "ldr.h"
#include "lmm.h"

using gibbs_without_u_screen_NS::gibbs_without_u_screen;
using gibbs_without_u_screen_NS::gibbs_without_u_screen_adaptive;
using VB_NS::VB;


auto setup(
    const vec &y,
    const mat &W,
    const mat &X,
    const mat *G,
    const double l_min,
    const double l_max,
    const size_t n_region,
    const bool do_rotate = true
    )
{
    struct Result {
        mat X;
        vec y;
        mat W;
        vec eigen_values;
        vec Wbeta;
        vec se_Wbeta;
        vec beta;
        double l_remle_null;
    } result;

    mat U(y.n_elem, y.n_elem); // eigen vectors
    if (do_rotate) {
         // eigen-decomposition and calculate trace_G
        arma::eig_sym(result.eigen_values, U, *G);
        result.eigen_values.transform( [](double val) { return val < 1e-10 ? 0 : val; } );

        // double trace_G = arma::mean(result.eigen_values);

        // rotate variables function
        result.W = U.t() * W;
        result.y = U.t() * y;
        result.X = U.t() * X;

        double logl_remle_H0;
        CalcLambda('R', result.eigen_values, result.W, result.y, l_min, l_max, n_region, result.l_remle_null, logl_remle_H0);
    }
    else {
        result.W = W;
        result.y = y;
        result.X = X;
        result.eigen_values = vec(y.n_elem, arma::fill::ones);
        U = mat(y.n_elem, y.n_elem, arma::fill::eye);
        result.l_remle_null = 1.0;
    }

    
    double vg_remle_null;
    double ve_remle_null;
    // double pve_null;
    // double pve_se_null;

    
    // CalcPve(result.eigen_values, result.UtW, result.Uty, result.l_remle_null, trace_G, pve_null, pve_se_null);
    CalcLmmVgVeBeta(result.eigen_values, result.W, result.y, result.l_remle_null, vg_remle_null, ve_remle_null, result.Wbeta, result.se_Wbeta);

    result.beta = result.l_remle_null * result.X.t() * ((U.t() * (y - (W * result.Wbeta))) / (result.eigen_values * result.l_remle_null + 1.0)) / result.X.n_cols;
    result.y -= arma::mean(result.y);

    return result;
}

Rcpp::List run_gibbs_without_u_screen_custom_kinship(
    vec &y,
    mat &W,
    mat &X,
    mat &G,
    size_t n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, &G, l_min, l_max, n_region);

    auto [alpha_vec, beta_vec, post_Ealpha, pheno_mean, pD1, pD2, DIC1, DIC2, BIC1, BIC2] = gibbs_without_u_screen(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null,
                                                            n_k,
                                                            w_step,
                                                            s_step,
                                                            show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
	                    Rcpp::Named("beta") = beta_vec,
	                    Rcpp::Named("post_Ealpha") = post_Ealpha,
	                    Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("pD1") = pD1,
                            Rcpp::Named("pD2") = pD2,
                            Rcpp::Named("DIC1") = DIC1,
                            Rcpp::Named("DIC2") = DIC2,
                            Rcpp::Named("BIC1") = BIC1,
                            Rcpp::Named("BIC2") = BIC2
                        )
                    );

}

Rcpp::List run_gibbs_without_u_screen(
    vec &y,
    mat &W,
    mat &X,
    size_t n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    mat G = (X * X.t()) / X.n_cols;
    return run_gibbs_without_u_screen_custom_kinship(y, W, X, G, n_k, w_step, s_step, l_min, l_max, n_region, show_progress);
}

Rcpp::List run_gibbs_without_u_screen_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, nullptr, l_min, l_max, n_region, false);

    auto [alpha_vec, beta_vec, post_Ealpha, pheno_mean, pD1, pD2, DIC1, DIC2, BIC1, BIC2] = gibbs_without_u_screen(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null,
                                                            n_k,
                                                            w_step,
                                                            s_step,
                                                            show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
	                    Rcpp::Named("beta") = beta_vec,
                        Rcpp::Named("post_Ealpha") = post_Ealpha,
                        Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("pD1") = pD1,
                            Rcpp::Named("pD2") = pD2,
                            Rcpp::Named("DIC1") = DIC1,
                            Rcpp::Named("DIC2") = DIC2,
                            Rcpp::Named("BIC1") = BIC1,
                            Rcpp::Named("BIC2") = BIC2
                        )
                    );

}

Rcpp::List run_gibbs_without_u_screen_adaptive_custom_kinship(
    vec &y,
    mat &W,
    mat &X,
    mat &G,
    size_t m_n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, &G, l_min, l_max, n_region);

    auto [alpha_vec, beta_vec, post_Ealpha, pheno_mean, pD1, pD2, DIC1, DIC2, BIC1, BIC2] = gibbs_without_u_screen_adaptive(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null,
                                                            m_n_k,
                                                            w_step,
                                                            s_step,
                                                            show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
                        Rcpp::Named("beta") = beta_vec,
                        Rcpp::Named("post_Ealpha") = post_Ealpha,
                        Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("pD1") = pD1,
                            Rcpp::Named("pD2") = pD2,
                            Rcpp::Named("DIC1") = DIC1,
                            Rcpp::Named("DIC2") = DIC2,
                            Rcpp::Named("BIC1") = BIC1,
                            Rcpp::Named("BIC2") = BIC2
                        )
                    );
}

Rcpp::List run_gibbs_without_u_screen_adaptive(
    vec &y,
    mat &W,
    mat &X,
    size_t m_n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    mat G = (X * X.t()) / X.n_cols;
    return run_gibbs_without_u_screen_adaptive_custom_kinship(y, W, X, G, m_n_k, w_step, s_step, l_min, l_max, n_region, show_progress);
}

Rcpp::List run_gibbs_without_u_screen_adaptive_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t m_n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, nullptr, l_min, l_max, n_region, false);

    auto [alpha_vec, beta_vec, post_Ealpha, pheno_mean, pD1, pD2, DIC1, DIC2, BIC1, BIC2] = gibbs_without_u_screen_adaptive(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null,
                                                            m_n_k,
                                                            w_step,
                                                            s_step,
                                                            show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
                        Rcpp::Named("beta") = beta_vec,
                        Rcpp::Named("post_Ealpha") = post_Ealpha,
                        Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("pD1") = pD1,
                            Rcpp::Named("pD2") = pD2,
                            Rcpp::Named("DIC1") = DIC1,
                            Rcpp::Named("DIC2") = DIC2,
                            Rcpp::Named("BIC1") = BIC1,
                            Rcpp::Named("BIC2") = BIC2
                        )
                    );
}

Rcpp::List run_VB_custom_kinship(
    vec &y,
    mat &W,
    mat &X,
    mat &G,
    size_t n_k,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, &G, l_min, l_max, n_region);

    auto [alpha_vec, beta_vec, pheno_mean, ELBO_vec] = VB(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null, n_k, show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
	                    Rcpp::Named("beta") = beta_vec,
                        Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("ELBO") = ELBO_vec
                        )
                    );
}

Rcpp::List run_VB(
    vec &y,
    mat &W,
    mat &X,
    size_t n_k,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    mat G = (X * X.t()) / X.n_cols;
    return run_VB_custom_kinship(y, W, X, G, n_k, l_min, l_max, n_region, show_progress);
}

Rcpp::List run_VB_no_kinship(
    arma::vec &y,
    arma::mat &W,
    arma::mat &X,
    size_t n_k,
    double l_min,
    double l_max,
    size_t n_region,
    bool show_progress
    )
{
    auto [UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null] = setup(y, W, X, nullptr, l_min, l_max, n_region, false);

    auto [alpha_vec, beta_vec, pheno_mean, ELBO_vec] = VB(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null, n_k, show_progress);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
	                    Rcpp::Named("beta") = beta_vec,
                        Rcpp::Named("pheno_mean") = pheno_mean,
                        Rcpp::Named("model_stats") = Rcpp::List::create(
                            Rcpp::Named("ELBO") = ELBO_vec
                        )
                    );
}

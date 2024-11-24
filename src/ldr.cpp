#include "ldr.h"
#include <algorithm>
#include <math.h>
#include "random_sampling.h"
#include "lmm.h"
#include "utils.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"
using arma::uword;

double sum_Elogvl2(const vec& vk, size_t k)
{
	return arma::accu(arma::log(1 - vk.head(k)));
}

// function 2: the first term of lambda_k
double sum_lambda_k(const mat& pik_beta, size_t k, size_t n_k)
{
	double slambda_k = 0;
	if ((k + 1) > (n_k - 1))
	{
		slambda_k = 0;
	}
	else
	{
		for (size_t j = k + 1; j < n_k - 1; j++)
		{
			slambda_k += arma::accu(pik_beta.col(j));
		}
	}
	return slambda_k;
}

// log(p(y|theta))
double logLike(const vec &D, const vec &y_res, double sigma2b, double sigma2e)
{
    vec H0 = (sigma2b * D) + 1;
    double a = arma::accu(arma::log(H0));
    H0 = y_res / H0;
    double b = arma::dot(H0, y_res);
    size_t n_idv = y_res.n_elem;
    double c = n_idv * (log(sigma2e) + log(2 * 3.1415926));
    double log_density = -0.5 * a - 0.5 * b / sigma2e - 0.5 * c;
    return log_density;
}

double log_h(const vec &D, const vec &y_res, double h, double sigma2e, double ae, double be)
{
    double sigma2b = h / (1 - h);
    vec H0 = (sigma2b * D) + 1;
    double a = arma::accu(arma::log(H0));
    H0 = y_res / H0;
    double b = arma::dot(H0, y_res);
    double c = (ae + 1) * log(sigma2b) + be / sigma2b;
    double log_density = -0.5 * a - 0.5 * b / sigma2e - c - 2 * log(1 - h);
    return log_density;
}

std::tuple<vec,vec> gibbs_without_u_screen(
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
    size_t s_step)
{

    get_output_file("out_folder/utx.txt") << UtX;
	get_output_file("out_folder/Uty.txt") << Uty;
	get_output_file("out_folder/UtW.txt") << UtW;
	get_output_file("out_folder/D.txt") << D;
	get_output_file("out_folder/Wbeta.txt") << Wbeta;
	get_output_file("out_folder/se_Wbeta.txt") << se_Wbeta;
	get_output_file("out_folder/beta.txt") << beta;
	get_output_file("out_folder/lambda.txt") << lambda;

    clock_t time_begin = clock();
    uword n_snp = UtX.n_cols; // n * p
    uword n_idv = Uty.n_elem;
    uword n_j = UtW.n_cols;
    vec x_col(n_idv);

    vec Utu(n_idv, arma::fill::zeros);
    vec Ue(n_idv);
    vec Ub(n_idv);

    vec bv0(n_idv, arma::fill::zeros); // for computation of alpha
    vec bv(n_idv, arma::fill::zeros);  // for computation of alpha
    vec V(n_idv, arma::fill::zeros);   // compute Utu
    vec M(n_idv, arma::fill::zeros);   // compute Utu
    vec snp_label(n_snp);

    vec Ealpha = Wbeta;                      // intercept
    vec post_Ealpha(n_j, arma::fill::zeros); // save intercept
    vec m_alpha = Wbeta;
    vec s2_alpha = se_Wbeta % se_Wbeta; // % is element wise multiplication see https://arma.sourceforge.net/docs.html#operators

    vec WEalpha(n_idv);
    double sigma2e = 0;
    double lambdax;
    vec XEbeta(n_idv); // G*beta
    vec Ebeta(n_snp);
    vec post_Ebeta(n_snp, arma::fill::zeros); // save beta
    mat Ebeta2k;
    mat B1;
    //// initial values for SNP and each normal
    //// component has the same initial values
    mat mik_beta = arma::repmat(beta, 1, n_k);
    mik_beta.row(0).zeros();

    vec llike(w_step + s_step, arma::fill::zeros);

    mat sik2_beta = arma::randu(n_snp, n_k, arma::distr_param(0, 1));
    mat pik_beta(n_snp, n_k);
    pik_beta.fill((double)1 / n_k);

    mat beta_beta(n_snp, n_k, arma::fill::zeros);
    mat gamma_beta(n_snp, n_k, arma::fill::zeros);
    mat post_gamma_beta = gamma_beta;
    gamma_beta += (double)1 / n_k;

    vec sigma2k(n_k);
    vec vk(n_k);
    vec Elogsigmak(n_k);
    vec Elogvk(n_k);
    vec sumElogvl(n_k);

    vec pikexp1(n_k);
    vec pikexp2(n_k);
    vec index;
    vec a_k(n_k);
    vec b_k(n_k);
    vec kappa_k(n_k);
    kappa_k.fill(n_snp * (double)1 / n_k);
    vec lambda_k = arma::randu(n_k, arma::distr_param(0, 1));

    // TODO: is xebeta just a zero vector with num elem UtX.n_row?
    XEbeta.zeros(); // set to zero
    // for (size_t i=0; i<n_snp; i++)   {
    // x_col   =  UtX.col(i);
    // XEbeta += x_col*beta(i);
    // }
    XEbeta = UtX * Ebeta;

    WEalpha = UtW * m_alpha;

    vec y_res = Uty - WEalpha - XEbeta;
    double sigma2e0 = (arma::dot(y_res, y_res)) / (n_idv - 1);
    double a_e = (2 * n_idv + n_snp) / 2;
    double b_e = (a_e - 1) * sigma2e0;

    double ak = 21;
    lambda = std::clamp(lambda, 0.01, 100.0);
    double bk0 = lambda * (ak - 1) / n_snp;

    vec bk(n_k);
    bk.at(0) = bk0;
    for (size_t i = 1; i < n_k; i++)
    {
        bk.at(i) = bk.at(i - 1) * 1.7 * sqrt(pow(i, i));
    }

    sik2_beta = arma::repmat(bk.t(), n_snp, 1) / ((ak - 1) * sigma2e0);

    double ae = 0.1;
    double be = 0.1;
    double a0 = 1;
    double b0 = 0.1;
    double a_lambda = a0 + n_k;
    double b_lambda = b0;
    double Elogsigmae, tx_ywx_res, xtxabk, A, post_Gn = 0;
    double sigma2b = 0.1;

    double post_llike = 0;
    double post_sigma2e = 0;
    double post_sigma2b = 0;

    vec sigma2bX = arma::randu(w_step + s_step + 1, arma::distr_param(0, 1));
    sigma2bX.at(0) = sigma2b;

    vec h = arma::randu(w_step + s_step + 1, arma::distr_param(0, 1));
    vec H0(n_idv);
    vec H(n_idv);

    ////initial values for a_k, b_k and sigma2k
    Ebeta2k = (mik_beta % mik_beta) + sik2_beta;

    a_k = arma::sum(pik_beta, 0).t() / 2 + ak;
    b_k = (arma::sum(Ebeta2k, 0).t() * (a_e / b_e) / 2) + bk;
    sigma2k = b_k / (a_k - 1);

    A = arma::dot(y_res, y_res);
    B1 = arma::repmat(1 / sigma2k, 1, n_snp); // repeat each row n_snp times
    B1.col(0).zeros();

    Ebeta2k = Ebeta2k % B1.t();
    double B = arma::accu(Ebeta2k);
    a_e = n_idv + n_snp * 0.1 + ae;
    b_e = (A + B + 2 * be) / 2;

    ////random seed
    RandomSampler rs(0);

    double pheno_mean; // TODO: see where it comes from and where it goes

    if (n_k == 1)
    {
        // WritelmmBeta(beta);
        pheno_mean = m_alpha(0);
    }
    // //  begin MCMC sampling
    else
    {
        ProgressBar progress_bar("MCMC Sampling", w_step + s_step);
        Rcpp::Rcout << progress_bar;
        for (size_t S = 0; S < (w_step + s_step); S++)
        {

            sigma2e = 1 / rs.gamma_sample(a_e, 1 / b_e);
            Elogsigmae = log(sqrt(sigma2e));

            // save Ebeta for the mixture normal component
            Ebeta = arma::sum(beta_beta % gamma_beta, 1);

            if (S > (w_step - 1))
            {
                post_Ebeta += Ebeta;
            }

            // sample sigma2k and compute related quantities
            for (size_t k = 0; k < n_k; k++)
            {
                if (k == 0)
                {
                    sigma2k.at(k) = 0;
                    Elogsigmak.at(k) = 0;
                    sumElogvl.at(k) = 0;
                }
                else
                {
                    sigma2k.at(k) = 1 / rs.gamma_sample(a_k.at(k), 1 / b_k.at(k));
                    Elogsigmak.at(k) = log(sqrt(sigma2k.at(k)));
                    sumElogvl.at(k) = sumElogvl.at(k - 1) + log(1 - vk.at(k - 1));
                }

                if (k == (n_k - 1))
                {
                    vk.at(k) = 0;
                    Elogvk.at(k) = 0;
                }
                else
                {
                    vk.at(k) = rs.beta_sample(kappa_k.at(k), lambda_k.at(k));
                    Elogvk.at(k) = log(vk.at(k));
                }
            }

            //////////////  sampling the mixture snp effects
            XEbeta.zeros(); // set Gbeta to zero first
                            // for (size_t i=0; i<n_snp; i++)   {
                            // x_col   =  UtX.col(i);
                            // XEbeta += x_col*Ebeta(i);
            //}
            XEbeta = UtX * Ebeta;

            y_res.zeros();
            y_res = Uty - WEalpha;

            H0 = (sigma2b * D) + 1;
            H0 = 1 / H0;

            //////////////////////////////////////
            // full sampling, i.e., sampling effects for all the snps
            // cout<<tp<<endl;
            for (size_t i = 0; i < n_snp; i++)
            {
                x_col = UtX.col(i);
                XEbeta -= x_col * Ebeta.at(i);

                H = x_col % H0;
                tx_ywx_res = arma::dot(H, y_res - XEbeta);

                for (size_t k = 0; k < n_k; k++)
                {

                    if (k == 0)
                    {
                        mik_beta.at(i, k) = 0;
                        sik2_beta.at(i, k) = 0;
                        pikexp1.at(k) = 0;
                    }

                    else
                    {
                        xtxabk = arma::dot(H, x_col) + 1 / sigma2k.at(k);
                        mik_beta.at(i, k) = tx_ywx_res / xtxabk;
                        sik2_beta.at(i, k) = sigma2e / xtxabk;
                        pikexp1.at(k) = mik_beta.at(i, k) * mik_beta.at(i, k) / (2 * sik2_beta.at(i, k)) +
                                        log(sqrt(sik2_beta.at(i, k))) - Elogsigmae - Elogsigmak.at(k);
                    }

                    beta_beta.at(i, k) = rs.gaussian_sample(sqrt(sik2_beta.at(i, k))) + mik_beta.at(i, k);
                    pikexp2.at(k) = Elogvk.at(k) + sumElogvl.at(k);
                }

                index = pikexp1 + pikexp2;
                index = arma::exp(index - arma::max(index));
                pik_beta.row(i) = index.t() / arma::accu(index);

                // multinomial sampling
                double mult_prob[n_k];
                unsigned int mult_no[n_k];

                for (size_t k = 0; k < n_k; k++)
                {
                    mult_prob[k] = pik_beta.at(i, k);
                }

                rs.multinomial_sample(n_k, 1, mult_prob, mult_no);
                for (size_t k = 0; k < n_k; k++)
                {
                    gamma_beta.at(i, k) = mult_no[k];
                }

                Ebeta(i) = arma::dot(beta_beta.row(i), gamma_beta.row(i));
                XEbeta += x_col * Ebeta.at(i);
            }

            //////////////////////////////////////

            if (S > (w_step - 1))
            {
                post_gamma_beta += gamma_beta;
            }
            //////////////  sampling the mixture snp effects

            WEalpha = UtW * Ealpha;

            y_res.zeros();
            y_res = Uty - XEbeta;
            for (size_t j = 0; j < n_j; j++)
            {
                WEalpha -= UtW.col(j) * Ealpha(j);
                H = UtW.col(j) % H0;
                m_alpha.at(j) = arma::dot(H, y_res - WEalpha) / arma::dot(H, UtW.col(j));
                s2_alpha.at(j) = sigma2e / arma::dot(H, UtW.col(j));
                Ealpha.at(j) = m_alpha.at(j) + rs.gaussian_sample(sqrt(s2_alpha.at(j)));
                WEalpha += UtW.col(j) * Ealpha.at(j);
            }

            if (S > (w_step - 1))
            {
                post_Ealpha += Ealpha;
            }

            a_lambda = a0 + n_k;
            b_lambda = b0 - sum_Elogvl2(vk, n_k - 1);
            lambdax = rs.gamma_sample(a_lambda, 1 / b_lambda);

            for (size_t k = 0; k < n_k - 1; k++)
            {
                kappa_k(k) = arma::accu(gamma_beta.col(k)) + 1;
                lambda_k(k) = sum_lambda_k(gamma_beta, k, n_k) + lambdax;
            }

            Ebeta2k = beta_beta % beta_beta % gamma_beta;
            a_k = arma::sum(gamma_beta, 0).t() / 2 + ak;
            b_k = arma::sum(Ebeta2k, 0).t() / (2 * sigma2e) + bk;

            /*y_res.setZero();
            y_res = Uty - XEbeta - WEalpha;
            for (size_t i=0; i<n_idv; i++) {
                V(i)  = sigma2b*D(i)/(sigma2b*D(i) + 1);
                M(i)  = y_res(i)*V(i);
                Utu(i)= M(i) + gsl_ran_gaussian(rs,sqrt(V(i)*sigma2e));
                if (D(i)  == 0) {
                    Ue(i) = 0;
                    Ub(i) = 0;
                    }
                else {
                    Ue(i) = Utu(i)/(sigma2b*D(i)); //for sigma2e
                    Ub(i) = Utu(i)/(sigma2e*D(i)); //for sigma2b
                }
                bv0(i) = y_res(i)*sigma2b/(sigma2b*D(i) + 1);
                }

            if (S>(w_step-1))   {bv += bv0;}
            */

            y_res.zeros();
            y_res = Uty - XEbeta - WEalpha;

            bv0 = (y_res * sigma2b) % H0;
            if (S > (w_step - 1))
            {
                bv += bv0;
            }

            H = y_res % H0;

            A = arma::dot(H, y_res);
            B1 = arma::repmat(1 / sigma2k.tail(n_k - 1), 1, n_snp); // repeat each row n_snp times
            Ebeta2k = Ebeta2k.cols(Ebeta2k.n_cols - n_k + 1, Ebeta2k.n_cols - 1) % B1.t();
            double B = arma::accu(Ebeta2k);
            double Gn = arma::accu(gamma_beta.cols(gamma_beta.n_cols - n_k + 1, gamma_beta.n_cols - 1));

            a_e = n_idv / 2 + Gn / 2 + ae;
            b_e = (A + B + 2 * be) / 2;

            /*double sigma2bX_new = sigma2bX(S) + gsl_ran_gaussian(rs,sqrt(1));
            double ratio1 = log_sigma2b(D,y_res,sigma2bX_new,sigma2e,ae,be);
            double ratio2 = log_sigma2b(D,y_res,sigma2bX(S), sigma2e,ae,be);
            double ratio  = exp(ratio1-ratio2);
            if (ratio>1) {ratio = 1;}
            else         {ratio = ratio;}
            if (sigma2bX_new<0) {ratio = 0;}
            */

            // double h_new  = gsl_rng_uniform(rs);
            double h_new = rs.beta_sample(2, 8);
            double ratio1 = log_h(D, y_res, h_new, sigma2e, ae, be) - log(rs.beta_pdf_val(h_new, 2, 8));
            double ratio2 = log_h(D, y_res, h(S), sigma2e, ae, be) - log(rs.beta_pdf_val(h(S), 2, 8));
            double ratio = exp(ratio1 - ratio2);
            if (ratio > 1)
            {
                ratio = 1;
            }
            else
            {
                ratio = ratio;
            }
            double u = rs.uniform_sample();
            if (u < ratio)
            {
                h(S + 1) = h_new;
            }
            else
            {
                h(S + 1) = h(S);
            }

            sigma2b = h(S + 1) / (1 - h(S + 1));

            if (S > (w_step - 1))
            {
                post_Gn += Gn;
            }
            ////////////////////////////////////

            double llike0 = logLike(D, y_res, sigma2b, sigma2e);
            llike(S) = llike0;

            if (S > (w_step - 1))
            {
                post_llike += llike0;
                post_sigma2e += sigma2e;
                post_sigma2b += sigma2b;
            }

            progress_bar.advance();
            Rcpp::Rcout << progress_bar;
        }
    }

    Rcpp::Rcout << std::endl << "MCMC sampling is finished" << std::endl;

    vec eigen_alpha = UtX.t() * (bv / s_step) / n_snp;

    return {eigen_alpha, post_Ebeta / s_step};
}

Rcpp::List run(
    vec &y,
    mat &W,
    mat &X,
    size_t n_k,
    size_t w_step,
    size_t s_step,
    double l_min,
    double l_max,
    size_t n_region
    )
{
    // Compute relatedness matrix...
    mat G = (X * X.t()) / X.n_cols;

    get_output_file("out_folder/G.txt") << G;

     // eigen-decomposition and calculate trace_G
    mat U(y.n_elem, y.n_elem); // eigen vectors
    vec eigen_values(y.n_elem);
    arma::eig_sym(eigen_values, U, G);
    eigen_values.transform( [](double val) { return val < 1e-10 ? 0 : val; } );

    double trace_G = arma::mean(eigen_values);
    // cout<<"Time for Eigen-Decomposition with GSL is "<<(clock()-time_start)/(double(CLOCKS_PER_SEC)*60.0)<<" mins"<<endl;

    get_output_file("out_folder/U.txt") << U;
    get_output_file("out_folder/W.txt") << W;

    mat UtW = U.t() * W;
    vec Uty = U.t() * y;
    mat UtX = U.t() * X;

    double l_remle_null;
    double logl_remle_H0;
    double pve_null;
    double pve_se_null;
    double vg_remle_null;
    double ve_remle_null;

    CalcLambda('R', arma_vec_to_gsl_vec(eigen_values).get(), arma_mat_to_gsl_mat(UtW).get(), arma_vec_to_gsl_vec(Uty).get(), l_min, l_max, n_region, l_remle_null, logl_remle_H0);

    CalcPve(arma_vec_to_gsl_vec(eigen_values).get(), arma_mat_to_gsl_mat(UtW).get(), arma_vec_to_gsl_vec(Uty).get(), l_remle_null, trace_G, pve_null, pve_se_null);

    gsl_vector *gsl_Wbeta = gsl_vector_alloc(W.n_cols);
    gsl_vector *gsl_se_Wbeta = gsl_vector_alloc(W.n_cols);

    CalcLmmVgVeBeta(arma_vec_to_gsl_vec(eigen_values).get(), arma_mat_to_gsl_mat(UtW).get(), arma_vec_to_gsl_vec(Uty).get(), l_remle_null, vg_remle_null, ve_remle_null, gsl_Wbeta, gsl_se_Wbeta);

    vec Wbeta = gsl_vec_to_arma_vec(gsl_Wbeta);
    vec se_Wbeta = gsl_vec_to_arma_vec(gsl_se_Wbeta);
    gsl_vector_free(gsl_Wbeta);
    gsl_vector_free(gsl_se_Wbeta);
    
    vec beta = l_remle_null * UtX.t() * ((U.t() * (y - (W * Wbeta))) / (eigen_values * l_remle_null + 1.0)) / UtX.n_cols;
    Uty -= arma::mean(Uty);

    auto [alpha_vec, beta_vec] = gibbs_without_u_screen(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, l_remle_null,
                                                            n_k,
                                                            w_step,
                                                            s_step);

    return Rcpp::List::create(
                        Rcpp::Named("alpha") = alpha_vec,
	                      Rcpp::Named("beta") = beta_vec
                    );

}


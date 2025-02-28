#include "ldr.h"
#include "random_sampling.h"
#include "utils.h"

#include <math.h>

using arma::uword;

double ELBO1(const vec& a_k, const vec& b_k, size_t n_k)
{
	double sum_ELBO1 = 0;
	for (size_t k = 1; k < n_k; k++)
	{
		sum_ELBO1 += lgamma(a_k.at(k)) - a_k.at(k) * log(b_k.at(k)) + a_k.at(k);
	}
	return sum_ELBO1;
}

double ELBO2(const vec& kappa_k, const vec& lambda_k, size_t n_k)
{
	double sum_ELBO2 = 0;
	for (size_t k = 0; k < n_k - 1; k++)
	{
		sum_ELBO2 += lgamma(kappa_k.at(k)) + lgamma(lambda_k.at(k)) -
					 lgamma(kappa_k.at(k) + lambda_k.at(k));
	}
	return sum_ELBO2;
}

double ELBO3(const mat& pik_beta, const mat& sik2_beta)
{
	double sum_ELBO3 = 0;
	mat pik_sik2;
	mat sik2_betax;
	sik2_betax = sik2_beta * 2 * 3.141593 * 2.718282;
	pik_sik2 = arma::log(pik_beta + 1e-10) - 0.5 * arma::log(sik2_betax + 1e-10);
	sum_ELBO3 = arma::accu(pik_sik2 % pik_beta);
	return sum_ELBO3;
}

double sum_b_lambda(const vec& lambda_k, const vec& kappa_k, size_t n_k)
{
	double sb_lambda = 0;
	for (size_t k = 0; k < n_k - 1; k++)
	{
		sb_lambda += R::digamma(lambda_k.at(k)) - R::digamma(kappa_k.at(k) + lambda_k.at(k));
	}
	return sb_lambda;
}

double sum_Elogvl(const vec& lambda_k, const vec& kappa_k, size_t k)
{
	double sumElogvl = 0;
	if (k == 0)
		sumElogvl = 0;
	else
	{
		for (size_t j = 0; j < k; j++)
		{
			sumElogvl += R::digamma(lambda_k.at(j)) - R::digamma(kappa_k.at(j) + lambda_k.at(j));
		}
	}
	return sumElogvl;
}

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

gibbs_without_u_screen_NS::Result gibbs_without_u_screen_NS::gibbs_without_u_screen(
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
    bool show_progress)
{
    gibbs_without_u_screen_NS::Result result;

    uword n_snp = UtX.n_cols; // n * p
    uword n_idv = Uty.n_elem;
    uword n_j = UtW.n_cols;

    vec bv(n_idv, arma::fill::zeros);  // for computation of alpha

    vec Ealpha = Wbeta;                      // intercept
    vec post_Ealpha(n_j, arma::fill::zeros); // save intercept
    vec m_alpha = Wbeta;
    vec s2_alpha = se_Wbeta % se_Wbeta; // % is element wise multiplication see https://arma.sourceforge.net/docs.html#operators

    vec WEalpha(n_idv);
    double sigma2e = 0;
    double lambdax;
    vec XEbeta(n_idv); // G*beta
    vec Ebeta(n_snp);
    result.beta = vec(n_snp, arma::fill::zeros); // save beta
    mat Ebeta2k;
    mat B1;
    //// initial values for SNP and each normal
    //// component has the same initial values
    mat mik_beta = arma::repmat(beta, 1, n_k);
    mik_beta.col(0).zeros();

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

    XEbeta = UtX * beta; // ToDo: check with dan if this is bug should be beta instead?
    WEalpha = UtW * m_alpha;

    vec y_res = Uty - WEalpha - XEbeta;
    double sigma2e0 = (arma::dot(y_res, y_res)) / (n_idv - 1);
    double a_e = (2 * n_idv + n_snp) / 2;
    double b_e = (a_e - 1) * sigma2e0;

    double ak = 21;
    lambda = std::clamp(lambda, 0.01, 100.0);

    vec bk(n_k);
    bk.at(0) = lambda * (ak - 1) / n_snp;
    for (size_t i = 1; i < n_k; i++)
    {
        bk.at(i) = bk.at(i - 1) * 1.7 * sqrt(pow(i, i));
    }

    sik2_beta = arma::repmat(bk.t(), n_snp, 1) / ((ak - 1) * sigma2e0);

    const double ae = 0.1;
    const double be = 0.1;
    const double a0 = 1;
    const double b0 = 0.1;

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

    RandomSampler rs;

    // //  begin MCMC sampling
    ProgressBar progress_bar("MCMC Sampling", w_step + s_step);
    if (show_progress) {
        Rcpp::Rcout << progress_bar;
    }

    for (size_t S = 0; S < (w_step + s_step); S++)
    {

        sigma2e = 1 / rs.gamma_sample(a_e, 1 / b_e);
        Elogsigmae = log(sqrt(sigma2e));

        // save Ebeta for the mixture normal component
        Ebeta = arma::sum(beta_beta % gamma_beta, 1);

        if (S > (w_step - 1))
        {
            result.beta += Ebeta;
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

        XEbeta = UtX * Ebeta;

        y_res = Uty - WEalpha;

        H0 = (sigma2b * D) + 1;
        H0 = 1 / H0;

        //////////////////////////////////////
        // full sampling, i.e., sampling effects for all the snps
        // cout<<tp<<endl;
        for (size_t i = 0; i < n_snp; i++)
        {
            XEbeta -= UtX.col(i) * Ebeta.at(i);

            H = UtX.col(i) % H0;
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
                    xtxabk = arma::dot(H, UtX.col(i)) + 1 / sigma2k.at(k);
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
            int mult_no[n_k];

            for (size_t k = 0; k < n_k; k++)
            {
                mult_prob[k] = pik_beta.at(i, k);
            }

            rs.multinomial_sample(n_k, 1, mult_prob, mult_no);
            for (size_t k = 0; k < n_k; k++)
            {
                gamma_beta.at(i, k) = mult_no[k];
            }

            Ebeta.at(i) = arma::dot(beta_beta.row(i), gamma_beta.row(i));
            XEbeta += UtX.col(i) * Ebeta.at(i);
        }

        //////////////////////////////////////

        if (S > (w_step - 1))
        {
            post_gamma_beta += gamma_beta;
        }
        //////////////  sampling the mixture snp effects

        WEalpha = UtW * Ealpha;

        y_res = Uty - XEbeta;
        for (size_t j = 0; j < n_j; j++)
        {
            WEalpha -= UtW.col(j) * Ealpha.at(j);
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
            kappa_k.at(k) = arma::accu(gamma_beta.col(k)) + 1;
            lambda_k.at(k) = sum_lambda_k(gamma_beta, k, n_k) + lambdax;
        }

        Ebeta2k = beta_beta % beta_beta % gamma_beta;
        a_k = arma::sum(gamma_beta, 0).t() / 2 + ak;
        b_k = arma::sum(Ebeta2k, 0).t() / (2 * sigma2e) + bk;

        y_res = Uty - XEbeta - WEalpha;

        if (S > (w_step - 1))
        {
            bv += (y_res * sigma2b) % H0;
        }

        H = y_res % H0;

        A = arma::dot(H, y_res);
        B1 = arma::repmat(1 / sigma2k.tail(n_k - 1), 1, n_snp); // repeat each row n_snp times
        Ebeta2k = Ebeta2k.cols(Ebeta2k.n_cols - n_k + 1, Ebeta2k.n_cols - 1) % B1.t();
        double B = arma::accu(Ebeta2k);
        double Gn = arma::accu(gamma_beta.cols(gamma_beta.n_cols - n_k + 1, gamma_beta.n_cols - 1));

        a_e = n_idv / 2 + Gn / 2 + ae;
        b_e = (A + B + 2 * be) / 2;

        double h_new = rs.beta_sample(2, 8);
        double ratio1 = log_h(D, y_res, h_new, sigma2e, ae, be) - log(rs.beta_pdf_val(h_new, 2, 8));
        double ratio2 = log_h(D, y_res, h(S), sigma2e, ae, be) - log(rs.beta_pdf_val(h(S), 2, 8));
        double ratio = exp(ratio1 - ratio2);
        if (ratio > 1)
        {
            ratio = 1;
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

        if (show_progress) {
            progress_bar.advance();
            Rcpp::Rcout << progress_bar;
        }
    }

    result.alpha = UtX.t() * (bv / s_step) / n_snp;
    result.beta /=  s_step;
    result.post_Ealpha = post_Ealpha;
    result.pheno_mean = post_Ealpha(0)/s_step;

    XEbeta = UtX * result.beta;
    WEalpha = UtW * post_Ealpha / s_step;
    y_res = Uty - XEbeta - WEalpha;

    double llike_hat = logLike(D, y_res, post_sigma2b / s_step, post_sigma2e / s_step);
    vec llike2 = llike.tail(s_step);
    llike2 -= arma::accu(llike2) / s_step;
    result.pD1 = 2 * (llike_hat - post_llike / s_step);
    result.pD2 = 2 * arma::dot(llike2, llike2) / (s_step - 1);
    if (result.pD1 < 0)
    {
        result.pD1 = 1;
    }
    result.DIC1 = -2 * llike_hat + 2 * result.pD1;
    result.DIC2 = -2 * llike_hat + 2 * result.pD2;
    result.BIC1 = -2 * llike_hat + log(n_idv) * result.pD1;
    result.BIC2 = -2 * llike_hat + log(n_idv) * result.pD2;

    return result;
}

VB_NS::Result VB_NS::VB(
    const mat &UtX,
    const vec &Uty,
    const mat &UtW,
    const vec &D,
    const vec &Wbeta,
    const vec &se_Wbeta,
    const vec &beta,
    double lambda,
    size_t n_k,
    bool show_progress)
{
    VB_NS::Result result;

    size_t n_snp = UtX.n_cols;
    size_t n_idv = Uty.n_elem;
    size_t n_j = UtW.n_cols;
    vec x_col(n_idv);

    vec Utu(n_idv, arma::fill::zeros);
    vec Ue(n_idv);
    vec Ub(n_idv);
    vec bv(n_idv, arma::fill::zeros); // for computation of alpha
    vec V(n_idv, arma::fill::zeros);	 // compute Utu
    vec M (n_idv, arma::fill::zeros);	 // compute Utu
    vec snp_label(n_snp);

    vec Ealpha = Wbeta;					// intercept
    vec m_alpha = Wbeta;
    vec s2_alpha = arma::square(se_Wbeta); // element wise square

    vec WEalpha = UtW * m_alpha;
    vec XEbeta = UtX * beta; // G*beta
    // vec Ebeta(n_snp);
    vec post_Ebeta(n_snp); // save beta
    mat Ebeta2k;
    mat B1;

    vec xtx = arma::sum(arma::square(UtX), 0).t();
    vec xty = UtX.t() * Uty;
    vec wtw = arma::sum(arma::square(UtW), 0).t();
    vec wty = UtW.t() * Uty;

    //// initial values for SNP and each normal
    //// component has the same initial values
    mat mik_beta = arma::repmat(beta, 1, n_k);
    mik_beta.col(0).zeros();

    mat sik2_beta = arma::randu(n_snp, n_k, arma::distr_param(0, 1));
    mat pik_beta(n_snp, n_k);
    pik_beta.fill((double)1 / n_k);

    // MatrixXd beta_beta  = mik_beta;
    mat beta_beta(n_snp, n_k, arma::fill::zeros);
    mat gamma_beta(n_snp, n_k, arma::fill::zeros);
    gamma_beta.fill((double)1 / n_k);
    mat post_gamma_beta(n_snp, n_k, arma::fill::zeros);
    mat post_pik_beta(n_snp, n_k, arma::fill::zeros);


    vec sigma2k(n_k);
    vec vk(n_k);
    vec Elogsigmak(n_k);
    vec Elogvk(n_k);
    vec sumElogvl(n_k);

    vec pikexp1(n_k);
    vec pikexp2(n_k);
    vec index;
    vec a_k = arma::randu(n_k, arma::distr_param(0, 1));
    vec b_k = arma::randu(n_k, arma::distr_param(0, 1));
    vec kappa_k(n_k);
    kappa_k.fill(n_snp * (double)1 / n_k);
    vec lambda_k = arma::randu(n_k, arma::distr_param(0, 1));

    vec y_res = Uty - WEalpha - XEbeta;
    double sigma2e0 = (arma::dot(y_res, y_res)) / (n_idv - 1);
    double a_e = (2 * n_idv + n_snp) / 2;
    double b_e = (a_e - 1) * sigma2e0;

    double ak = 21;
    lambda = std::clamp(lambda, 0.01, 100.0);

    vec bk(n_k);
    bk.at(0) = lambda * (ak - 1) / n_snp;
    for (size_t i = 1; i < n_k; i++)
    {
        bk.at(i) = bk.at(i - 1) * 1.7 * sqrt(pow(i, i));
    }

    sik2_beta = arma::repmat(bk.t(), n_snp, 1) / ((ak - 1) * sigma2e0);

    double ae = 0.1;
    double be = 0.1;
    double a_b = 0.1;
    double b_b = 0.1;
    // double a0   = 400;
    // double b0   = 40;
    double a0 = 1;
    double b0 = 0.1;
    double a_lambda = a0 + n_k;
    double b_lambda = b0;
    double Elogsigmae, tx_ywx_res, xtxabk, A;

    ////initial values for a_k, b_k and sigma2k
    Ebeta2k = arma::square(mik_beta) + sik2_beta;
    result.beta = arma::sum(mik_beta % pik_beta, 1);

    a_k = arma::sum(pik_beta, 0).t() / 2 + ak;
    b_k = (arma::sum(Ebeta2k, 0).t() * (a_e / b_e) / 2) + bk;
    sigma2k = b_k / (a_k - 1);

    A = arma::dot(y_res, y_res);
    B1 = arma::repmat(1 / sigma2k, 1, n_snp);
    B1.col(0).zeros();

    Ebeta2k = Ebeta2k % B1.t();
    double B = arma::accu(Ebeta2k);
    a_e = n_idv + n_snp * 0.1 + ae;
    b_e = (A + B + 2 * be) / 2;

    size_t int_step = 0;
    size_t max_step = n_idv * sqrt(10);
    double delta = 10;
    vec ELBO(max_step, arma::fill::zeros);

    // WritelmmBeta(beta);
    if (n_k == 1)
    {
        result.pheno_mean = m_alpha(0);
    }
    /////////////////// variational bayesian/////////////
    else
    {
        while ((int_step < max_step) && (delta > 1e-7))
        {

            Elogsigmae = 0.5 * (log(b_e) - R::digamma(a_e));
            //////////////  sampling the mixture snp effects
            XEbeta = UtX * result.beta;

            lambda_k(n_k - 1) = 0;
            double ab_e = a_e / b_e; // E(sigma2e-2)
            double ba_e = b_e / a_e; // 1/E(sigma2e-2)

            y_res = Uty - WEalpha - Utu;
            for (size_t i = 0; i < n_snp; i++)
            {
                x_col = UtX.col(i);
                XEbeta -= x_col * result.beta.at(i);
                tx_ywx_res = arma::dot(x_col, y_res - XEbeta);

                for (size_t k = 0; k < n_k; k++)
                {
                    if (k == 0)
                    {
                        mik_beta.at(i, k) = 0;
                        sik2_beta.at(i, k) = 0;
                        pikexp1.at(k) = 0;
                        Elogsigmak.at(k) = 0;
                    }
                    else
                    {
                        xtxabk = xtx.at(i) + a_k.at(k) / b_k.at(k);
                        mik_beta.at(i, k) = tx_ywx_res / xtxabk;
                        sik2_beta.at(i, k) = ba_e / xtxabk;
                        Elogsigmak.at(k) = 0.5 * (log(b_k.at(k)) - R::digamma(a_k.at(k)));
                        pikexp1.at(k) = pow(mik_beta.at(i, k), 2) / (2 * sik2_beta.at(i, k)) +
                                        log(sqrt(sik2_beta.at(i, k))) - Elogsigmae - Elogsigmak.at(k);
                        // pikexp1(k)     =-pow(mik_beta(i,k),2)*(a_k(k)/b_k(k))*ab_e/2- Elogsigmae - Elogsigmak(k);
                    }

                    // if (k == (n_k-1))   {
                    // vk(k)	  = 1;
                    // Elogvk(k) = 0;
                    // }
                    // else   {
                    Elogvk.at(k) = R::digamma(kappa_k.at(k)) - R::digamma(kappa_k.at(k) + lambda_k.at(k));
                    //}

                    sumElogvl.at(k) = sum_Elogvl(lambda_k, kappa_k, k);
                    pikexp2.at(k) = Elogvk.at(k) + sumElogvl.at(k);
                }

                index = pikexp1 + pikexp2;
                index = arma::exp(index - arma::max(index));
                pik_beta.row(i) = index.t() / arma::accu(index);

                result.beta.at(i) = arma::dot(mik_beta.row(i), pik_beta.row(i));
                XEbeta += x_col * result.beta.at(i);
            }
            //////////////  sampling the mixture snp effects

            WEalpha = UtW * Ealpha;

            y_res = Uty - XEbeta - Utu;
            for (size_t j = 0; j < n_j; j++)
            {
                WEalpha -= UtW.col(j) * Ealpha.at(j);
                Ealpha.at(j) = 1 / wtw.at(j) * arma::dot(UtW.col(j), y_res - WEalpha);
                s2_alpha.at(j) = ba_e / wtw.at(j);
                WEalpha += UtW.col(j) * Ealpha.at(j);
            }

            a_lambda = a0 + n_k;
            b_lambda = b0 - sum_b_lambda(lambda_k, kappa_k, n_k);

            for (size_t k = 0; k < n_k - 1; k++)
            {
                kappa_k.at(k) = arma::accu(pik_beta.col(k)) + 1;
                lambda_k.at(k) = sum_lambda_k(pik_beta, k, n_k) + a_lambda / b_lambda;
            }

            Ebeta2k = (arma::square(mik_beta) + sik2_beta) % pik_beta;
            a_k = arma::sum(pik_beta, 0).t() / 2 + ak;
            b_k = arma::sum(Ebeta2k, 0).t() * ab_e / 2 + bk;

            y_res = Uty - XEbeta - WEalpha;
            double ab = a_b / b_b;
            for (size_t i = 0; i < n_idv; i++)
            {
                if (D.at(i) == 0)
                {
                    V.at(i) = 0;
                    Utu.at(i) = 0;
                    Ue.at(i) = 0;
                    Ub.at(i) = 0;
                }
                else
                {
                    double abD = ab / D.at(i);
                    V.at(i) = ba_e / (abD + 1);
                    Utu.at(i) = y_res(i) / (abD + 1);
                    Ue.at(i) = (Utu.at(i) * Utu.at(i) + V.at(i)) * abD;			// for sigma2e
                    Ub.at(i) = (Utu.at(i) * Utu.at(i) + V.at(i)) * ab_e / D.at(i); // for sigma2b
                }
                // bv(i) = y_res(i)/(D(i) + a_b/b_b);
            }

            vec VarBeta = arma::sum(Ebeta2k, 1) - arma::square(result.beta);
            A = arma::dot(y_res - Utu, y_res - Utu) + arma::dot(wtw, s2_alpha) +
                    arma::accu(V) + arma::dot(xtx, VarBeta);
            B1 = arma::repmat(a_k / b_k, 1, n_snp);
            B1.col(0).zeros();
            double B = arma::accu(Ebeta2k % B1.t());
            double Gn = arma::accu(pik_beta.cols(pik_beta.n_cols - n_k + 1, pik_beta.n_cols - 1));

            // mixture_no=Gn;

            a_e = n_idv + Gn / 2 + ae;
            b_e = (A + B + arma::accu(Ue)) / 2 + be;

            a_b = n_idv / 2 + ae;
            b_b = arma::accu(Ub) / 2 + be;

            int_step++;
            vec ab_k = a_k / b_k;

            ELBO.at(int_step) = (lgamma(a_e) - a_e * log(b_e) +
                            lgamma(a_b) - a_b * log(b_b) + a_b +
                            ELBO1(a_k, b_k, n_k) +
                            ELBO2(kappa_k, lambda_k, n_k) -
                            ELBO3(pik_beta, sik2_beta) +
                            0.5 * arma::accu(arma::log(s2_alpha)) +
                            0.5 * arma::accu(arma::log(V + 1e-10)) +
                            lgamma(a_lambda) - a_lambda * log(b_lambda)) +
                            a_lambda -
                            be * (arma::accu(ab_k.tail(n_k - 1)) + ab + a_lambda / b_lambda);

            ////////////////////////////////////
            delta = abs((ELBO.at(int_step) - ELBO.at(int_step - 1)) / ELBO.at(int_step));

            // if ((int_step+1)%10==0) {cout<<int_step+1<<" "<<setprecision(5)<<delta<<" "<<ELBO(int_step)<<endl;}
        }
    }

    result.pheno_mean = Ealpha(0);

    y_res = Uty - XEbeta - WEalpha;
    bv = y_res / (D + (a_b / b_b));

    result.alpha = UtX.t() * bv / n_snp;
    result.ELBO = ELBO.head(int_step);

    Rcpp::Rcout << "variational bayes is finished" << std::endl;

    return result;
}

gibbs_without_u_screen_NS::Result gibbs_without_u_screen_NS::gibbs_without_u_screen_adaptive(
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
    bool show_progress)
{
    Rcpp::Rcout << "Now start to adaptively select nk..." << std::endl;
    double min_dic = 10e100;
    size_t n_k = 4;
    double sp = 0.1; // todo: add to func param

    for (size_t j = 0; j < (m_n_k - 1); j++)
    {
        Rcpp::Rcout << "nk == " << j + 2 << std::endl;
        // cLdr.Gibbs_without_u_screen_dic0(UtX, y0, W0, D, Wbeta0, se_Wbeta0, beta, snp_no, lambda, j + 2);
        auto [_unused1, _unused2, _unused3, unused4, _unused5, _unused6, DIC1, _unused7, _unused8, _unused9] = gibbs_without_u_screen(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, lambda,
                                                            j + 2,
                                                            w_step * sp,
                                                            s_step * sp,
                                                            show_progress);
        Rcpp::Rcout << "DIC is " << DIC1 << std::endl;
        if (DIC1 < min_dic)
        {
            min_dic = DIC1;
            n_k = j + 2;
        }
    }

    Rcpp::Rcout << "The adaptive selection procedure is finished nk == " << n_k << " was selcted with DIC " << min_dic << std::endl;
    Rcpp::Rcout << "Now start to MCMC sampling with adaptively selected nk..." << std::endl;
    // cLdr.Gibbs_without_u_screen_dic1(UtX, y0, W0, D, Wbeta0, se_Wbeta0, beta, snp_no, lambda, n_k);

    return gibbs_without_u_screen(UtX, Uty, UtW, eigen_values, Wbeta, se_Wbeta, beta, lambda,
                                                            n_k,
                                                            w_step,
                                                            s_step,
                                                            show_progress);
}


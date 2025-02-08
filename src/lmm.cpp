/*
	Genome-wide Efficient Mixed Model Association (GEMMA)
	Copyright (C) 2011  Xiang Zhou

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "lmm.h"
#include "root_solver.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <cstring>
#include <limits>
#include <vector>

using namespace std;

// map a number 1-(n_cvt+2) to an index between 0 and [(n_c+2)^2+(n_c+2)]/2-1
size_t GetabIndex(const size_t a, const size_t b, const size_t n_cvt)
{
	if (a > n_cvt + 2 || b > n_cvt + 2 || a <= 0 || b <= 0)
	{
		Rcpp::Rcout << "error in GetabIndex." << endl;
		return 0;
	}
	size_t index;
	size_t l, h;
	if (b > a)
	{
		l = a;
		h = b;
	}
	else
	{
		l = b;
		h = a;
	}

	size_t n = n_cvt + 2;
	index = (2 * n - l + 2) * (l - 1) / 2 + h - l;

	return index;
}

void CalcPab(const size_t n_cvt, const size_t e_mode, const vec &Hi_eval, const mat &Uab, const vec &ab, mat &Pab)
{
	size_t index_ab, index_aw, index_bw, index_ww;
	double p_ab;
	double ps_ab, ps_aw, ps_bw, ps_ww;

	for (size_t p = 0; p <= n_cvt + 1; ++p)
	{
		for (size_t a = p + 1; a <= n_cvt + 2; ++a)
		{
			for (size_t b = a; b <= n_cvt + 2; ++b)
			{
				index_ab = GetabIndex(a, b, n_cvt);
				if (p == 0)
				{
					p_ab = arma::dot(Hi_eval, Uab.col(index_ab));
					if (e_mode != 0)
					{
						p_ab = ab.at(index_ab) - p_ab;
					}
					Pab.at(0, index_ab) = p_ab;
				}
				else
				{
					index_aw = GetabIndex(a, p, n_cvt);
					index_bw = GetabIndex(b, p, n_cvt);
					index_ww = GetabIndex(p, p, n_cvt);

					ps_ab = Pab.at(p - 1, index_ab);
					ps_aw = Pab.at(p - 1, index_aw);
					ps_bw = Pab.at(p - 1, index_bw);
					ps_ww = Pab.at(p - 1, index_ww);

					p_ab = ps_ab - ps_aw * ps_bw / ps_ww;
					Pab.at(p, index_ab) = p_ab;
				}
			}
		}
	}
}

void CalcPPab(const size_t n_cvt, const size_t e_mode, const vec &HiHi_eval, const mat &Uab, const vec &ab, const mat &Pab, mat &PPab)
{
	size_t index_ab, index_aw, index_bw, index_ww;
	double p2_ab;
	double ps2_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww;

	for (size_t p = 0; p <= n_cvt + 1; ++p)
	{
		for (size_t a = p + 1; a <= n_cvt + 2; ++a)
		{
			for (size_t b = a; b <= n_cvt + 2; ++b)
			{
				index_ab = GetabIndex(a, b, n_cvt);
				if (p == 0)
				{
					p2_ab = arma::dot(HiHi_eval, Uab.col(index_ab));
					if (e_mode != 0)
					{
						p2_ab = p2_ab - ab.at(index_ab) + 2.0 * Pab.at(0, index_ab);
					}
					PPab.at(0, index_ab) = p2_ab;
				}
				else
				{
					index_aw = GetabIndex(a, p, n_cvt);
					index_bw = GetabIndex(b, p, n_cvt);
					index_ww = GetabIndex(p, p, n_cvt);

					ps2_ab = PPab.at(p - 1, index_ab);
					ps_aw = Pab.at(p - 1, index_aw);
					ps_bw = Pab.at(p - 1, index_bw);
					ps_ww = Pab.at(p - 1, index_ww);
					ps2_aw = PPab.at(p - 1, index_aw);
					ps2_bw = PPab.at(p - 1, index_bw);
					ps2_ww = PPab.at(p - 1, index_ww);

					p2_ab = ps2_ab + ps_aw * ps_bw * ps2_ww / (ps_ww * ps_ww);
					p2_ab -= (ps_aw * ps2_bw + ps_bw * ps2_aw) / ps_ww;
					PPab.at(p, index_ab) = p2_ab;
				}
			}
		}
	}
}

void CalcPPPab(const size_t n_cvt, const size_t e_mode, const vec &HiHiHi_eval, const mat &Uab, const vec &ab, const mat &Pab, const mat &PPab, mat &PPPab)
{
	size_t index_ab, index_aw, index_bw, index_ww;
	double p3_ab;
	double ps3_ab, ps_aw, ps_bw, ps_ww, ps2_aw, ps2_bw, ps2_ww, ps3_aw, ps3_bw, ps3_ww;

	for (size_t p = 0; p <= n_cvt + 1; ++p)
	{
		for (size_t a = p + 1; a <= n_cvt + 2; ++a)
		{
			for (size_t b = a; b <= n_cvt + 2; ++b)
			{
				index_ab = GetabIndex(a, b, n_cvt);
				if (p == 0)
				{
					p3_ab = arma::dot(HiHiHi_eval, Uab.col(index_ab));
					if (e_mode != 0)
					{
						p3_ab = ab.at(index_ab) - p3_ab + 3.0 * PPab.at(0, index_ab) - 3.0 * Pab.at(0, index_ab);
					}
					PPPab.at(0, index_ab) = p3_ab;
				}
				else
				{
					index_aw = GetabIndex(a, p, n_cvt);
					index_bw = GetabIndex(b, p, n_cvt);
					index_ww = GetabIndex(p, p, n_cvt);

					ps3_ab = PPPab.at(p - 1, index_ab);
					ps_aw = Pab.at(p - 1, index_aw);
					ps_bw = Pab.at(p - 1, index_bw);
					ps_ww = Pab.at(p - 1, index_ww);
					ps2_aw = PPab.at(p - 1, index_aw);
					ps2_bw = PPab.at(p - 1, index_bw);
					ps2_ww = PPab.at(p - 1, index_ww);
					ps3_aw = PPPab.at(p - 1, index_aw);
					ps3_bw = PPPab.at(p - 1, index_bw);
					ps3_ww = PPPab.at(p - 1, index_ww);

					p3_ab = ps3_ab - ps_aw * ps_bw * ps2_ww * ps2_ww / (ps_ww * ps_ww * ps_ww);
					p3_ab -= (ps_aw * ps3_bw + ps_bw * ps3_aw + ps2_aw * ps2_bw) / ps_ww;
					p3_ab += (ps_aw * ps2_bw * ps2_ww + ps_bw * ps2_aw * ps2_ww + ps_aw * ps_bw * ps3_ww) / (ps_ww * ps_ww);

					PPPab.at(p, index_ab) = p3_ab;
				}
			}
		}
	}
}

double LogL_f(double l, FUNC_PARAM &params)
{
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
	}

	double f = 0.0, logdet_h = 0.0;
	size_t index_yy;

	mat Pab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	logdet_h = arma::accu(arma::log(arma::abs(params.eval * l + 1.0)));

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);

	double c = 0.5 * (double)ni_test * (log((double)ni_test) - log(2 * M_PI) - 1.0);

	index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_yy);
	f = c - 0.5 * logdet_h - 0.5 * (double)ni_test * log(P_yy);

	return f;
}

double LogL_dev1(double l, void *p)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
	}

	double dev1 = 0.0, trace_Hi = 0.0;
	size_t index_yy;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;
	trace_Hi = arma::accu(Hi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);

	double trace_HiK = ((double)ni_test - trace_Hi) / l;

	index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);

	double P_yy = Pab.at(nc_total, index_yy);
	double PP_yy = PPab.at(nc_total, index_yy);
	double yPKPy = (P_yy - PP_yy) / l;
	dev1 = -0.5 * trace_HiK + 0.5 * (double)ni_test * yPKPy / P_yy;

	return dev1;
}

double LogL_dev2(double l, void *p)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
	}

	double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
	size_t index_yy;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	mat PPPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);
	vec HiHiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;
	HiHiHi_eval = HiHi_eval % Hi_eval;

	trace_Hi = arma::accu(Hi_eval);
	trace_HiHi = arma::accu(HiHi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
		trace_HiHi = 2 * trace_Hi + trace_HiHi - (double)ni_test;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);
	CalcPPPab(n_cvt, params.e_mode, HiHiHi_eval, params.Uab, params.ab, Pab, PPab, PPPab);

	double trace_HiKHiK = ((double)ni_test + trace_HiHi - 2 * trace_Hi) / (l * l);

	index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_yy);
	double PP_yy = PPab.at(nc_total, index_yy);
	double PPP_yy = PPPab.at(nc_total, index_yy);

	double yPKPy = (P_yy - PP_yy) / l;
	double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

	dev2 = 0.5 * trace_HiKHiK - 0.5 * (double)ni_test * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);

	return dev2;
}

void LogL_dev12(double l, void *p, double *dev1, double *dev2)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
	}

	double trace_Hi = 0.0, trace_HiHi = 0.0;
	size_t index_yy;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	mat PPPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);
	vec HiHiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;
	HiHiHi_eval = HiHi_eval % Hi_eval;

	trace_Hi = arma::accu(Hi_eval);
	trace_HiHi = arma::accu(HiHi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
		trace_HiHi = 2 * trace_Hi + trace_HiHi - (double)ni_test;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);
	CalcPPPab(n_cvt, params.e_mode, HiHiHi_eval, params.Uab, params.ab, Pab, PPab, PPPab);

	double trace_HiK = ((double)ni_test - trace_Hi) / l;
	double trace_HiKHiK = ((double)ni_test + trace_HiHi - 2 * trace_Hi) / (l * l);

	index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);

	double P_yy = Pab.at(nc_total, index_yy);
	double PP_yy = PPab.at(nc_total, index_yy);
	double PPP_yy = PPPab.at(nc_total, index_yy);

	double yPKPy = (P_yy - PP_yy) / l;
	double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

	*dev1 = -0.5 * trace_HiK + 0.5 * (double)ni_test * yPKPy / P_yy;
	*dev2 = 0.5 * trace_HiKHiK - 0.5 * (double)ni_test * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);
}

double LogRL_f(double l, FUNC_PARAM &params)
{
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	double df;
	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
		df = (double)ni_test - (double)n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
		df = (double)ni_test - (double)n_cvt - 1.0;
	}

	double f = 0.0, logdet_h = 0.0, logdet_hiw = 0.0, d;
	size_t index_ww;

	mat Pab(n_cvt + 2, n_index);
	mat Iab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec v_temp(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	logdet_h = arma::accu(arma::log(arma::abs(params.eval * l + 1.0)));

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	v_temp.fill(1.0);
	CalcPab(n_cvt, params.e_mode, v_temp, params.Uab, params.ab, Iab);

	// calculate |WHiW|-|WW|
	logdet_hiw = 0.0;
	for (size_t i = 0; i < nc_total; ++i)
	{
		index_ww = GetabIndex(i + 1, i + 1, n_cvt);
		d = Pab.at(i, index_ww);
		logdet_hiw += log(d);
		d = Iab.at(i, index_ww);
		logdet_hiw -= log(d);
	}
	index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_ww);

	double c = 0.5 * df * (log(df) - log(2 * M_PI) - 1.0);
	f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy);

	return f;
}

double LogRL_dev1(double l, void *p)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	double df;
	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
		df = (double)ni_test - (double)n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
		df = (double)ni_test - (double)n_cvt - 1.0;
	}

	double dev1 = 0.0, trace_Hi = 0.0;
	size_t index_ww;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;

	trace_Hi = arma::accu(Hi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);

	// calculate tracePK and trace PKPK
	double trace_P = trace_Hi;
	double ps_ww, ps2_ww;
	for (size_t i = 0; i < nc_total; ++i)
	{
		index_ww = GetabIndex(i + 1, i + 1, n_cvt);
		ps_ww = Pab.at(i, index_ww);
		ps2_ww = PPab.at(i, index_ww);
		trace_P -= ps2_ww / ps_ww;
	}
	double trace_PK = (df - trace_P) / l;

	// calculate yPKPy, yPKPKPy
	index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_ww);
	double PP_yy = PPab.at(nc_total, index_ww);
	double yPKPy = (P_yy - PP_yy) / l;

	dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;

	return dev1;
}

double LogRL_dev2(double l, void *p)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	double df;
	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
		df = (double)ni_test - (double)n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
		df = (double)ni_test - (double)n_cvt - 1.0;
	}

	double dev2 = 0.0, trace_Hi = 0.0, trace_HiHi = 0.0;
	size_t index_ww;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	mat PPPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);
	vec HiHiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;
	HiHiHi_eval = HiHi_eval % Hi_eval;

	trace_Hi = arma::accu(Hi_eval);
	trace_HiHi = arma::accu(HiHi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
		trace_HiHi = 2 * trace_Hi + trace_HiHi - (double)ni_test;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);
	CalcPPPab(n_cvt, params.e_mode, HiHiHi_eval, params.Uab, params.ab, Pab, PPab, PPPab);

	// calculate tracePK and trace PKPK
	double trace_P = trace_Hi, trace_PP = trace_HiHi;
	double ps_ww, ps2_ww, ps3_ww;
	for (size_t i = 0; i < nc_total; ++i)
	{
		index_ww = GetabIndex(i + 1, i + 1, n_cvt);
		ps_ww = Pab.at(i, index_ww);
		ps2_ww = PPab.at(i, index_ww);
		ps3_ww = PPPab.at(i, index_ww);
		trace_P -= ps2_ww / ps_ww;
		trace_PP += ps2_ww * ps2_ww / (ps_ww * ps_ww) - 2.0 * ps3_ww / ps_ww;
	}
	double trace_PKPK = (df + trace_PP - 2.0 * trace_P) / (l * l);

	// calculate yPKPy, yPKPKPy
	index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_ww);
	double PP_yy = PPab.at(nc_total, index_ww);
	double PPP_yy = PPPab.at(nc_total, index_ww);
	double yPKPy = (P_yy - PP_yy) / l;
	double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

	dev2 = 0.5 * trace_PKPK - 0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);

	return dev2;
}

void LogRL_dev12(double l, void *p, double *dev1, double *dev2)
{
	FUNC_PARAM &params = *(FUNC_PARAM *)p;
	size_t n_cvt = params.n_cvt;
	size_t ni_test = params.ni_test;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	double df;
	size_t nc_total;
	if (params.calc_null == true)
	{
		nc_total = n_cvt;
		df = (double)ni_test - (double)n_cvt;
	}
	else
	{
		nc_total = n_cvt + 1;
		df = (double)ni_test - (double)n_cvt - 1.0;
	}

	double trace_Hi = 0.0, trace_HiHi = 0.0;
	size_t index_ww;

	mat Pab(n_cvt + 2, n_index);
	mat PPab(n_cvt + 2, n_index);
	mat PPPab(n_cvt + 2, n_index);
	vec Hi_eval(params.eval.n_elem);
	vec HiHi_eval(params.eval.n_elem);
	vec HiHiHi_eval(params.eval.n_elem);

	if (params.e_mode == 0)
	{
		Hi_eval = 1.0 / (params.eval * l + 1.0);
	}
	else
	{
		Hi_eval = 1.0 - (1.0 / (params.eval * l + 1.0));
	}

	HiHi_eval = Hi_eval % Hi_eval;
	HiHiHi_eval = HiHi_eval % Hi_eval;

	trace_Hi = arma::accu(Hi_eval);
	trace_HiHi = arma::accu(HiHi_eval);

	if (params.e_mode != 0)
	{
		trace_Hi = (double)ni_test - trace_Hi;
		trace_HiHi = 2 * trace_Hi + trace_HiHi - (double)ni_test;
	}

	CalcPab(n_cvt, params.e_mode, Hi_eval, params.Uab, params.ab, Pab);
	CalcPPab(n_cvt, params.e_mode, HiHi_eval, params.Uab, params.ab, Pab, PPab);
	CalcPPPab(n_cvt, params.e_mode, HiHiHi_eval, params.Uab, params.ab, Pab, PPab, PPPab);

	// calculate tracePK and trace PKPK
	double trace_P = trace_Hi, trace_PP = trace_HiHi;
	double ps_ww, ps2_ww, ps3_ww;
	for (size_t i = 0; i < nc_total; ++i)
	{
		index_ww = GetabIndex(i + 1, i + 1, n_cvt);
		ps_ww = Pab.at(i, index_ww);
		ps2_ww = PPab.at(i, index_ww);
		ps3_ww = PPPab.at(i, index_ww);
		trace_P -= ps2_ww / ps_ww;
		trace_PP += ps2_ww * ps2_ww / (ps_ww * ps_ww) - 2.0 * ps3_ww / ps_ww;
	}
	double trace_PK = (df - trace_P) / l;
	double trace_PKPK = (df + trace_PP - 2.0 * trace_P) / (l * l);

	// calculate yPKPy, yPKPKPy
	index_ww = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(nc_total, index_ww);
	double PP_yy = PPab.at(nc_total, index_ww);
	double PPP_yy = PPPab.at(nc_total, index_ww);
	double yPKPy = (P_yy - PP_yy) / l;
	double yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (l * l);

	*dev1 = -0.5 * trace_PK + 0.5 * df * yPKPy / P_yy;
	*dev2 = 0.5 * trace_PKPK - 0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy);
}

void CalcUab(const mat &UtW, const vec &Uty, mat &Uab)
{
	size_t index_ab;
	size_t n_cvt = UtW.n_cols;

	vec u_a(Uty.n_elem);

	for (size_t a = 1; a <= n_cvt + 2; ++a)
	{
		if (a == n_cvt + 1)
		{
			continue;
		}

		if (a == n_cvt + 2)
		{
			u_a = Uty;
		}
		else
		{
			u_a = UtW.col(a - 1);
		}

		for (size_t b = a; b >= 1; --b)
		{
			if (b == n_cvt + 1)
			{
				continue;
			}

			index_ab = GetabIndex(a, b, n_cvt);

			if (b == n_cvt + 2)
			{
				Uab.col(index_ab) = Uty;
			}
			else
			{
				Uab.col(index_ab) = UtW.col(b - 1);
			}

			Uab.col(index_ab) %= u_a; // element wise multiplication
		}
	}
}

void CalcLambda(const char func_name, FUNC_PARAM &params, const double l_min, const double l_max, const size_t n_region, double &lambda, double &logf)
{
	if (func_name != 'R' && func_name != 'L' && func_name != 'r' && func_name != 'l')
	{
		Rcpp::Rcout << "func_name only takes 'R' or 'L': 'R' for log-restricted likelihood, 'L' for log-likelihood." << endl;
		return;
	}

	vector<pair<double, double>> lambda_lh;

	// evaluate first order derivates in different intervals
	double lambda_l, lambda_h, lambda_interval = log(l_max / l_min) / (double)n_region;
	double dev1_l, dev1_h, logf_l, logf_h;

	for (size_t i = 0; i < n_region; ++i)
	{
		lambda_l = l_min * exp(lambda_interval * i);
		lambda_h = l_min * exp(lambda_interval * (i + 1.0));

		if (func_name == 'R' || func_name == 'r')
		{
			dev1_l = LogRL_dev1(lambda_l, &params);
			dev1_h = LogRL_dev1(lambda_h, &params);
		}
		else
		{
			dev1_l = LogL_dev1(lambda_l, &params);
			dev1_h = LogL_dev1(lambda_h, &params);
		}

		if (dev1_l * dev1_h <= 0)
		{
			lambda_lh.push_back(make_pair(lambda_l, lambda_h));
		}
	}

	// if derivates do not change signs in any interval
	if (lambda_lh.empty())
	{
		if (func_name == 'R' || func_name == 'r')
		{
			logf_l = LogRL_f(l_min, params);
			logf_h = LogRL_f(l_max, params);
		}
		else
		{
			logf_l = LogL_f(l_min, params);
			logf_h = LogL_f(l_max, params);
		}

		if (logf_l >= logf_h)
		{
			lambda = l_min;
			logf = logf_l;
		}
		else
		{
			lambda = l_max;
			logf = logf_h;
		}
	}
	else
	{
		// if derivates change signs
		double l;
		root_solver::function fx;
		root_solver::function dfx;
		root_solver::fdf_function fdfx;

		if (func_name == 'R' || func_name == 'r')
		{
			fx = LogRL_dev1;
			dfx = LogRL_dev2;
			fdfx = LogRL_dev12;
		}
		else
		{
			fx = LogL_dev1;
			dfx = LogL_dev2;
			fdfx = LogL_dev12;
		}

		logf = -std::numeric_limits<double>::infinity();

		for (const auto &[lambda_l, lambda_h] : lambda_lh)
		{
			l = solve_root_brent(fx, &params, lambda_l, lambda_h);
			l = solve_root_newton(fx, dfx, fdfx, &params, l, l_min, l_max);

			if (func_name == 'R' || func_name == 'r')
			{
				logf_l = LogRL_f(l, params);
			}
			else
			{
				logf_l = LogL_f(l, params);
			}

			if (logf < logf_l)
			{
				logf = logf_l;
				lambda = l;
			}
		}

		if (func_name == 'R' || func_name == 'r')
		{
			logf_l = LogRL_f(l_min, params);
			logf_h = LogRL_f(l_max, params);
		}
		else
		{
			logf_l = LogL_f(l_min, params);
			logf_h = LogL_f(l_max, params);
		}

		if (logf_l > logf)
		{
			lambda = l_min;
			logf = logf_l;
		}
		if (logf_h > logf)
		{
			lambda = l_max;
			logf = logf_h;
		}
	}
}

// calculate lambda in the null model
void CalcLambda(const char func_name, const vec &eval, const mat &UtW, const vec &Uty, const double l_min, const double l_max, const size_t n_region, double &lambda, double &logl_H0)
{
	if (func_name != 'R' && func_name != 'L' && func_name != 'r' && func_name != 'l')
	{
		Rcpp::Rcout << "func_name only takes 'R' or 'L': 'R' for log-restricted likelihood, 'L' for log-likelihood." << endl;
		return;
	}

	size_t n_cvt = UtW.n_cols;
	size_t ni_test = UtW.n_rows;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	mat Uab(ni_test, n_index, arma::fill::zeros);
	vec ab(n_index);

	CalcUab(UtW, Uty, Uab);

	FUNC_PARAM params = {true, ni_test, n_cvt, eval, Uab, ab, 0};
	CalcLambda(func_name, params, l_min, l_max, n_region, lambda, logl_H0);
}

// obtain REMLE estimate for PVE using lambda_remle
void CalcPve(const vec &eval, const mat &UtW, const vec &Uty, const double lambda, const double trace_G, double &pve, double &pve_se)
{
	size_t n_cvt = UtW.n_cols;
	size_t ni_test = UtW.n_rows;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	mat Uab(ni_test, n_index, arma::fill::zeros);
	vec ab(n_index);

	CalcUab(UtW, Uty, Uab);

	FUNC_PARAM params = {true, ni_test, n_cvt, eval, Uab, ab, 0};

	double se = sqrt(-1.0 / LogRL_dev2(lambda, &params));

	pve = trace_G * lambda / (trace_G * lambda + 1.0);
	pve_se = trace_G / ((trace_G * lambda + 1.0) * (trace_G * lambda + 1.0)) * se;

	return;
}

// obtain REML estimate for Vg and Ve using lambda_remle
// obtain beta and se(beta) for coefficients
// ab is not used when e_mode==0
void CalcLmmVgVeBeta(const vec &eval, const mat &UtW, const vec &Uty, const double lambda, double &vg, double &ve, vec &beta, vec &se_beta)
{
	size_t n_cvt = UtW.n_cols;
	size_t ni_test = UtW.n_rows;
	size_t n_index = (n_cvt + 2 + 1) * (n_cvt + 2) / 2;

	mat Uab(ni_test, n_index, arma::fill::zeros);
	vec ab(n_index);
	mat Pab(n_cvt + 2, n_index);
	vec Hi_eval(eval.n_elem);
	mat HiW(eval.n_elem, UtW.n_cols);
	mat WHiW(UtW.n_cols, UtW.n_cols);
	vec WHiy(UtW.n_cols);
	mat Vbeta(UtW.n_cols, UtW.n_cols);

	CalcUab(UtW, Uty, Uab);

	Hi_eval = 1.0 / (eval * lambda + 1.0);

	// calculate beta
	HiW = UtW;
	HiW %= arma::repmat(Hi_eval, 1, UtW.n_cols);
	WHiW = HiW.t() * UtW;
	WHiy = HiW.t() * Uty;

	beta = arma::solve(WHiW, WHiy);
	Vbeta = arma::inv(WHiW);

	// calculate vg and ve
	CalcPab(n_cvt, 0, Hi_eval, Uab, ab, Pab);

	size_t index_yy = GetabIndex(n_cvt + 2, n_cvt + 2, n_cvt);
	double P_yy = Pab.at(n_cvt, index_yy);

	ve = P_yy / (double)(ni_test - n_cvt);
	vg = ve * lambda;

	// with ve, calculate se(beta)
	Vbeta *= ve;

	// obtain se_beta
	se_beta = arma::sqrt(Vbeta.diag());
}

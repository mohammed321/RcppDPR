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

#ifndef __LMM_H__
#define __LMM_H__

#include <RcppArmadillo.h>

using arma::vec;
using arma::mat;

class FUNC_PARAM
{

public:
	bool calc_null;
	size_t ni_test;
	size_t n_cvt;
	const vec &eval;
	const mat &Uab;
	const vec &ab;
	size_t e_mode;
};

void CalcLambda (const char func_name, FUNC_PARAM &params, const double l_min, const double l_max, const size_t n_region, double &lambda, double &logf);
void CalcLambda (const char func_name, const vec &eval, const mat &UtW, const vec &Uty, const double l_min, const double l_max, const size_t n_region, double &lambda, double &logl_H0);
void CalcPve (const vec &eval, const mat &UtW, const vec &Uty, const double lambda, const double trace_G, double &pve, double &pve_se);
void CalcLmmVgVeBeta (const vec &eval, const mat &UtW, const vec &Uty, const double lambda, double &vg, double &ve, vec &beta, vec &se_beta);

#endif



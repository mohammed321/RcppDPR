#include <iostream>

namespace root_solver {
    typedef double (*function)(double x, void* params);
    typedef void (*fdf_function)(double x, void* params, double *f, double *df);
}

double solve_root_brent(root_solver::function fx, void* params, double x_lower, double x_upper, size_t max_iter = 100, double epsabs = 0, double epsrel = 1e-1);
double solve_root_newton(root_solver::function fx, root_solver::function dfx, root_solver::fdf_function fdfx, void* params, double x_guess, const double x_min, const double x_max, size_t max_iter = 100, double epsabs = 0, double epsrel = 1e-5);
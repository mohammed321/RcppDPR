
#include "root_solver.h"
#include <algorithm>

#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

void my_error_handler (
                        const char * reason,
                        const char * file,
                        int line,
                        int gsl_errno
                    )
{
    gsl_stream_printf ("ERROR", file, line, reason);
}

void set_error_handler() {
    gsl_set_error_handler(&my_error_handler);
}

void reset_error_handler_default() {
    gsl_set_error_handler(nullptr);
}

double solve_root_brent(root_solver::function fx, void* params, double x_lower, double x_upper, size_t max_iter, double epsabs, double epsrel) {

    set_error_handler();

    const gsl_root_fsolver_type *T_f = gsl_root_fsolver_brent;
    gsl_root_fsolver *s_f = gsl_root_fsolver_alloc(T_f);

    gsl_function F;
    F.function = fx;
    F.params = params;

    gsl_root_fsolver_set(s_f, &F, x_lower, x_upper);
    double root;

    for (size_t i = 0; i < max_iter; i++) {
        gsl_root_fsolver_iterate(s_f);
        root = gsl_root_fsolver_root(s_f);
        x_lower = gsl_root_fsolver_x_lower(s_f);
        x_upper = gsl_root_fsolver_x_upper(s_f);
        if (gsl_root_test_interval(x_lower, x_upper, epsabs, epsrel) != GSL_CONTINUE) break;
    }

    gsl_root_fsolver_free(s_f);

    reset_error_handler_default();

    return root;
}

double solve_root_newton(root_solver::function fx, root_solver::function dfx, root_solver::fdf_function fdfx, void* params, double x_guess, const double x_min, const double x_max, size_t max_iter, double epsabs, double epsrel) {

    set_error_handler();

    const gsl_root_fdfsolver_type *T_fdf = gsl_root_fdfsolver_newton;
    gsl_root_fdfsolver *s_fdf = gsl_root_fdfsolver_alloc(T_fdf);

    gsl_function_fdf FDF;
    FDF.f = fx;
    FDF.df = dfx;
    FDF.fdf = fdfx;
    FDF.params = params;

    gsl_root_fdfsolver_set(s_fdf, &FDF, x_guess);
    double root;

    for (size_t i = 0; i < max_iter; i++) {
        gsl_root_fdfsolver_iterate(s_fdf);
        root = gsl_root_fdfsolver_root(s_fdf);
        if (gsl_root_test_delta(root, x_guess, 0, 1e-5) != GSL_CONTINUE || root <= x_min || root >= x_max) break;
        x_guess = root;
    }

    root = x_guess;
    root = std::clamp(root, x_min, x_max);

    gsl_root_fdfsolver_free(s_fdf);

    reset_error_handler_default();

    return root;
}

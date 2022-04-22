import unittest
from Opt_HW1.src.unconstrained_min import *
from Opt_HW1.test.examples import *


class TestMinMethods(unittest.TestCase):
    def test_quad_1(self):
        func = f_quad_1
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([1, 1])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                                           max_iter, 'GDC')
        x_history_newton, f_history_newton, converge_newton = minimize(func, x0, step_len_newton, c1, rho, obj_tol,
                                                                    param_tol, max_iter, 'Newton')
        plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd
                  ,step_len_newton)

    def test_quad_2(self):
        func = f_quad_2
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([1, 1])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                   max_iter, 'GDC')
        x_history_newton, f_history_newton, converge_newton = minimize(func, x0, step_len_newton, c1, rho, obj_tol,
                                            param_tol, max_iter, 'Newton')
        plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd
        , step_len_newton)

    def test_quad_3(self):
        func = f_quad_3
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([1, 1])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                                           max_iter, 'GDC')
        x_history_newton, f_history_newton, converge_newton = minimize(func, x0, step_len_newton, c1, rho, obj_tol,
                                                                    param_tol, max_iter, 'Newton')
        plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd
                  , step_len_newton)

    def test_rosenbrock(self):
        func = rosenbrock_func
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        max_iter_gdc = 10000
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter_newton = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([-1,2])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                                           max_iter_gdc, 'GDC')
        x_history_newton, f_history_newton, converge_newton = minimize(func, x0, step_len_newton, c1, rho, obj_tol,
                                                                   param_tol, max_iter_newton, 'Newton')
        plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd
                  , step_len_newton)


    def test_linear_func(self):
        func = linear_func
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([1, 1])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                                        max_iter, 'GDC')
        plot_results(x_history_gcd, f_history_gcd, None, None, func, step_len_gcd, step_len_newton)

    def test_exp_func(self):
        func = exp_func
        step_len_gcd = 'wolfe'
        step_len_newton = 'wolfe'
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c1 = 0.01
        rho = 0.5
        x0 = np.array([1, 1])

        x_history_gcd, f_history_gcd, converge_gcd = minimize(func, x0, step_len_gcd, c1, rho, obj_tol, param_tol,
                                                            max_iter, 'GDC')
        x_history_newton, f_history_newton, converge_newton = minimize(func, x0, step_len_newton, c1, rho, obj_tol,
                                                                 param_tol, max_iter, 'Newton')
        plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd
               , step_len_newton)


if __name__ == '__main__':
    unittest.main()

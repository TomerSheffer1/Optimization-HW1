import matplotlib.pyplot as plt
from Opt_HW1.test.examples import *


def print_run_status(min_func,itr,x_prev,f_val):
    if min_func == 'GDC':
        print('Gradient Descent iter num: ' + str(itr))
    elif min_func == 'Newton':
        print('Newton method iter num: ' + str(itr))

    print('Current location: x_' + str(itr) + ' = ' + str(x_prev))
    print('Current objective value: f_' + str(itr) + ' = ' + str(f_val) + '\n')


def plot_results(x_history_gcd, f_history_gcd, x_history_newton, f_history_newton, func, step_len_gcd, step_len_newton):
    if func == linear_func:
        v = np.linspace(-100,100,100)
    else:
        v = np.linspace(-3, 3, 100)  # contour lines map
    x1,x2 = np.meshgrid(v,v)

    X = np.vstack([x1.flatten(), x2.flatten()])
    F,_,_ = func(X,0,1)

    fig,(ax1,ax2) = plt.subplots(1,2)

    cs = ax1.contour(x1,x2,F,15)
    ax1.clabel(cs,inline=True,fontsize=10)
    scatter_min_path(ax1, x_history_gcd, 'b', 'GD')
    plot_min_process(ax2, f_history_gcd, step_len_gcd, 'GD',func)
    if x_history_newton is not None:
        scatter_min_path(ax1, x_history_newton, 'r', 'Newton')
        plot_min_process(ax2, f_history_newton, step_len_newton, 'Newton',func)

    fig.suptitle(function_def(func))
    plt.show()


def scatter_min_path(ax,x_history,color,method):
    x1_h_gcd = x_history[:, 0]
    x2_h_gcd = x_history[:, 1]
    ax.scatter(x1_h_gcd, x2_h_gcd, marker='x', c=color, label=method)
    annotate_path(ax,x_history,color)
    ax.set_title('Minimization paths')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()


def annotate_path(ax,x_history,color):
    for i in range(1, x_history.shape[0]):
        ax.annotate('', xy=x_history[i], xytext=x_history[i - 1],arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1})


def plot_min_process(ax,f_history,step_len,method,func):
    last_idx = np.max(np.nonzero(f_history))

    if step_len == 'wolfe':
        label = method + ' - Wolfe cond with backtracking'
    else:
        label = method + ' - ' + r'$\alpha$' + ' = ' + str(step_len)
    if last_idx == 0:
        ax.plot(np.arange(last_idx + 2), f_history[0:last_idx + 2], marker='o', label=label)
    else:
        ax.plot(np.arange(last_idx), f_history[0:last_idx], marker='o', label=label)

    if func is not linear_func:
        ax.set_yscale('log')
        ax.set_xscale('log')

    ax.set_ylabel('Function Value')
    ax.set_title('Function value Vs. Iterations')
    ax.set_xlabel('Iteration [#]')
    ax.legend()
    ax.grid()


def function_def(func):
    if func == f_quad_1:
        func_name = 'Quadratic form - circle contour lines'
    elif func == f_quad_2:
        func_name = 'Quadratic form - aligned ellipses contour lines'
    elif func == f_quad_3:
        func_name = 'Quadratic form - rotated ellipses contour lines'
    elif func == rosenbrock_func:
        func_name = 'Rosenbrock function'
    elif func == linear_func:
        func_name = 'Linear function'
    elif func == exp_func:
        func_name = 'Exponential function'

    return func_name

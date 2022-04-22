from Opt_HW1.src.utils import *


def minimize(f,x0,step_len,c1,rho,obj_tol,param_tol,max_iter,min_func):
    if step_len == 'wolfe':
        alpha = 1.05
    else:
        alpha = step_len

    if min_func == 'GDC':  # gradient descent
        itr = 0
        f_val, f_grad, _ = f(x0, 0)
        x_prev = x0
        x_history = np.zeros((max_iter,2))
        f_history = np.zeros(max_iter)
        converge = False
        while itr < max_iter and not converge:
            # Print run info
            print_run_status(min_func,itr,x_prev,f_val)

            # Save path and values
            x_history[itr,:] = x_prev
            f_history[itr] = f_val

            # Determine new step length (if needed)
            if step_len == 'wolfe':
                wolfe = check_wolfe_cond(x_prev,-f_grad,alpha,f_val,f_grad,c1,f)
                if not wolfe:
                    alpha = calc_new_steplen(alpha,rho,x_prev,-f_grad,f_val,f_grad,c1,f)

            # Update step
            x_new = x_prev - alpha * f_grad
            f_val, f_grad, _ = f(x_new,0)

            # Check convergence
            if itr > 0:
                converge = check_convergence(x_new,x_history[itr,:],f_val,f_history[itr-1],obj_tol,param_tol)

            # Updates for next iteration
            x_prev = x_new
            itr += 1

        print('Converged = ' + str(converge))
        print('#######################\n')
        return x_history, f_history, converge
    elif min_func == 'Newton':  # Newton Method
        k = 0
        f_k, fk_grad, fk_hessian = f(x0,1)
        x_k = x0
        x_history = np.zeros((max_iter, 2))
        f_history = np.zeros(max_iter)
        converge = False

        while k < max_iter and not converge:
            # Print run info
            print_run_status(min_func, k, x_k, f_k)

            # Save path and values
            x_history[k, :] = x_k
            f_history[k] = f_k

            # Determine new search direction
            p_k = np.linalg.solve(fk_hessian,-fk_grad)

            # Determine new step length (if needed)
            if step_len == 'wolfe':
                wolfe = check_wolfe_cond(x_k,p_k,alpha,f_k,fk_grad,c1,f)
                if not wolfe:
                    alpha = calc_new_steplen(alpha,rho,x_k,p_k,f_k,fk_grad,c1,f)
                    # print('New alpha - ' + str(alpha))
            # Update step
            x_next = x_k + alpha * p_k
            f_k, fk_grad, fk_hessian = f(x_next, 1)

            # Check convergence
            if k > 0:
                converge = check_convergence(x_next, x_history[k,:], f_k, f_history[k-1], obj_tol, param_tol)

            # Updates for next iteration
            x_k = x_next
            k += 1

        print('Converged = ' + str(converge))
        print('#######################\n')
        return x_history, f_history, converge


def check_convergence(x_new,x_prev,f_new,f_prev,delta,eps):
    step = np.linalg.norm(x_new - x_prev)
    df = abs(f_new - f_prev)
    return step < eps or df < delta


def check_wolfe_cond(x_k,p_k,step_len,fk,fk_grad,c,f):
    lhs,_,_ = f(x_k + step_len*p_k,0)
    rhs = fk + c*step_len*fk_grad.dot(p_k)
    return lhs <= rhs


def calc_new_steplen(alpha,mu,x_k,p_k,f_k,fk_grad,c,f):
    ls, _, _ = f(x_k + alpha * p_k, 0)
    rs = f_k + c*alpha*fk_grad.dot(p_k)
    while ls > rs:
        alpha = alpha * mu
        ls, _, _ = f(x_k + alpha * p_k, 0)
        rs = f_k + c*alpha*fk_grad.dot(p_k)
    return alpha

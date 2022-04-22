import numpy as np


def f_quad_1(x,eval_hessian,contour_lines=0):
    Q = np.array([[1,0],[0,1]])
    f = x.T.dot(Q).dot(x)
    if contour_lines:
        f = np.diag(f).reshape(100, 100)
    g = (Q+Q.T).dot(x)
    if eval_hessian:
        h = Q + Q.T
    else:
        h = None
    return f,g,h


def f_quad_2(x,eval_hessian,contour_lines=0):
    Q = np.array([[1,0],[0,100]])
    f = x.T.dot(Q).dot(x)
    if contour_lines:
        f = np.diag(f).reshape(100, 100)
    g = (Q+Q.T).dot(x)
    if eval_hessian:
        h = Q + Q.T
    else:
        h = None
    return f,g,h


def f_quad_3(x,eval_hessian,contour_lines=0):
    q1 = np.array([[np.sqrt(3)/2,-0.5],[0.5,np.sqrt(3)/2]])
    q2 = np.array([[100,0],[0,1]])
    Q = q1.T.dot(q2).dot(q1)
    f = x.T.dot(Q).dot(x)
    if contour_lines:
        f = np.diag(f).reshape(100, 100)
    g = (Q+Q.T).dot(x)
    if eval_hessian:
        h = Q + Q.T
    else:
        h = None

    return f,g,h


def rosenbrock_func(x0,eval_hessian,contour_lines=0):
    x1, x2 = x0
    if contour_lines:
        x1,x2 = np.meshgrid(np.linspace(-0.5,2,100),np.linspace(-1.5,4,100))
    f = 100*(x2-x1**2)**2 + (1-x1)**2
    g = np.array([400*x1**3 - 400*x1*x2 + 2*x1 -2, -200*x1**2 + 200*x2])
    if eval_hessian:
        h = np.array([[1200*x1**2 - 400*x2 + 2, -400*x1],[-400*x1, 200]])
    else:
        h = None

    return f,g,h


def linear_func(x0,use_hessian,contour_lines=0):
    np.random.seed(42)

    if contour_lines:
        a = np.random.random(2)
        x1, x2 = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
        f = a[0] * x1 + a[1] * x2
        g = None
        h = None
    else:
        a = np.random.random(x0.shape)
        f = a.dot(x0)
        g = a
        h = 0

    return f,g,h


def exp_func(x0,eval_hessian,contour_lines=0):
    if contour_lines:
        x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    else:
        x1, x2 = x0
    f = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1), 3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1) ])
    if eval_hessian:
        h = np.array([[np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1) , 3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1)],[ 3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1), 9*np.exp(x1 + 3*x2 - 0.1) + 9*np.exp(x1 - 3*x2 - 0.1)]])
    else:
        h = None

    return f,g,h

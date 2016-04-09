import numpy as np
from l1tf import l1tf
from matplotlib import pylab as plt


def test_l1tf(n=50, seed=4242, iter_max=1000, rho=1, lam=1.0,
              prompt=False, tol=1e-8, verbose=True, do_plot=True):
    """
    :param n: dimension of vector
    :param seed: random seed
    :param iter_max: maximum number of iterations
    :param rho: the ADMM step parameter
    :param lam: the problem's l1 regularization parameter
    :param prompt: show plots and print stuff at each step
                (default False)
    :param tol: Stop if max change between steps is lower than this
                times the max value of y
    :param verbose: Print stuff (default False)
    :param do_plot: Make a plot (default True)
    :return:
    """
    if seed is not None:
        np.random.seed(seed)
    y = np.cumsum(np.random.randn(n))
    x = l1tf(y, iter_max=iter_max, rho=rho, lam=lam, tol=tol,
             prompt=prompt, verbose=verbose)
    if do_plot:
        plt.clf()
        plt.plot(x, 'ro-')
        plt.plot(y, 'bo-')
        plt.show()
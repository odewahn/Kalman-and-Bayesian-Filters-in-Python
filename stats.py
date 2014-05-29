import math
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spln

_two_pi = 2*math.pi


def gaussian(x, mean, var):
    """returns normal distribution for x given a gaussian with the specified
    mean and variance. All must be scalars
    """
    return math.exp((-0.5*(x-mean)**2)/var) / math.sqrt(_two_pi*var)

def mul (a_mu, a_var, b_mu, b_var):
    m = (a_var*b_mu + b_var*a_mu) / (a_var + b_var)
    v = 1. / (1./a_var + 1./b_var)
    return (m, v)

def add (a_mu, a_var, b_mu, b_var):
    return (a_mu+b_mu, a_var+b_var)


def multivariate_gaussian(x, mu, cov):
    """ This is designed to work the same as scipy.stats.multivariate_normal
    which is not available before version 0.14. You may either pass in a
    multivariate set of data:
       multivariate_gaussian (array([1,1]), array([3,4]), eye(2)*1.4)
       multivariate_gaussian (array([1,1,1]), array([3,4,5]), 1.4)
    or unidimensional data:
       multivariate_gaussian(1, 3, 1.4)

    In the multivariate case if cov is a scalar it is interpreted as eye(n)*cov

    The function gaussian() implements the 1D (univariate)case, and is much
    faster than this function.
    """

    # force all to numpy.array type
    x   = np.array(x, copy=False, ndmin=1)
    mu  = np.array(mu,copy=False, ndmin=1)

    nx = len(mu)
    cov = _to_cov(cov, nx)

    norm_coeff = nx*math.log(2*math.pi) + np.linalg.slogdet(cov)[1]

    err = x - mu
    if (sp.issparse(cov)):
        numerator = spln.spsolve(cov, err).T.dot(err)
    else:
        numerator = np.linalg.solve(cov, err).T.dot(err)

    return math.exp(-0.5*(norm_coeff + numerator))


def norm_plot(mean, var):
    min_x = mean - var * 1.5
    max_x = mean + var * 1.5

    xs = np.arange(min_x, max_x, 0.1)
    ys = [gaussian(x,mean,var) for x in xs]
    plt.plot(xs,ys)


def sigma_ellipse(cov, x=0, y=0, sigma=1, num_pts=100):
    """ Takes a 2D covariance matrix and generates an ellipse showing the
    contour plot at the specified sigma value. Ellipse is centered at (x,y).
    num_pts specifies how many discrete points are used to generate the
    ellipse.

    Returns a tuple containing the ellipse,x, and y, in that order.
    The ellipse is a 2D numpy array with shape (2, num_pts). Row 0 contains the
    x components, and row 1 contains the y coordinates
    """
    cov = np.asarray(cov)

    L = linalg.cholesky(cov)
    t = np.linspace(0, _two_pi, num_pts)
    unit_circle = np.array([np.cos(t), np.sin(t)])

    ellipse = sigma * L.dot(unit_circle)
    ellipse[0] += x
    ellipse[1] += y
    return (ellipse,x,y)

def sigma_ellipses(cov, x=0, y=0, sigma=[1,2], num_pts=100):
    cov = np.asarray(cov)

    L = linalg.cholesky(cov)
    t = np.linspace(0, _two_pi, num_pts)
    unit_circle = np.array([np.cos(t), np.sin(t)])

    e_list = []
    for s in sigma:
        ellipse = s * L.dot(unit_circle)
        ellipse[0] += x
        ellipse[1] += y
        e_list.append (ellipse)
    return (e_list,x,y)

def plot_covariance_ellipse (cov, x=0, y=0, sigma=1,title=None, axis_equal=True):
    """ Plots the ellipse of the provided 2x2 covariance matrix.
    """
    e = sigma_ellipse (cov, x, y, sigma)
    plot_sigma_ellipse(e, title, axis_equal)


def plot_sigma_ellipse(ellipse, title=None, axis_equal=True):
    """ plots the ellipse produced from sigma_ellipse."""

    if axis_equal:
        plt.axis('equal')

    e = ellipse[0]
    x = ellipse[1]
    y = ellipse[2]

    plt.plot(e[0], e[1],c='b')
    plt.scatter(x,y,marker='+') # mark the center
    if title is not None:
        plt.title (title)

def plot_sigma_ellipses(ellipses,title=None,axis_equal=True,x_lim=None,y_lim=None):
    """ plots the ellipse produced from sigma_ellipse."""

    if x_lim is not None:
        axis_equal = False
        plt.xlim(x_lim)

    if y_lim is not None:
        axis_equal = False
        plt.ylim(y_lim)

    if axis_equal:
        plt.axis('equal')

    for ellipse in ellipses:
        es = ellipse[0]
        x = ellipse[1]
        y = ellipse[2]

        for e in es:
            plt.plot(e[0], e[1], c='b')

        plt.scatter(x,y,marker='+') # mark the center
    if title is not None:
        plt.title (title)


def _to_cov(x,n):
    """ If x is a scalar, returns a covariance matrix generated from it
    as the identity matrix multiplied by x. The dimension will be nxn.
    If x is already a numpy array then it is returned unchanged.
    """
    try:
        x.shape
        if type(x) != np.ndarray:
            x = np.asarray(x)[0]
        return x
    except:
        return np.eye(n) * x


if __name__ == '__main__':
    from scipy.stats import norm

    # test conversion of scalar to covariance matrix
    x  = multivariate_gaussian(np.array([1,1]), np.array([3,4]), np.eye(2)*1.4)
    x2 = multivariate_gaussian(np.array([1,1]), np.array([3,4]), 1.4)
    assert x == x2

    # test univarate case
    rv = norm(loc = 1., scale = np.sqrt(2.3))
    x2 = multivariate_gaussian(1.2, 1., 2.3)
    x3 = gaussian(1.2, 1., 2.3)

    assert rv.pdf(1.2) == x2
    assert abs(x2- x3) < 0.00000001

    cov = np.array([[1,1],
                    [1,1.1]])

    sigma = [1,1]
    ev = sigma_ellipses(cov, x=2, y=2, sigma=sigma)
    plot_sigma_ellipses([ev], axis_equal=True,x_lim=[0,4],y_lim=[0,15])
    #isct = plt.Circle((2,2),1,color='b',fill=False)
    #plt.figure().gca().add_artist(isct)
    plt.show()
    print "all tests passed"

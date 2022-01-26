import numpy as _np
from scipy.spatial.distance import pdist as _pdist, squareform as _squareform


def _svgd_kernel(theta, h = -1):
    sq_dist = _pdist(theta)
    pairwise_dists = _squareform(sq_dist)**2
    if h < 0: # if h < 0, using median trick
        h = _np.median(pairwise_dists)  
        h = _np.sqrt(0.5 * h / _np.log(theta.shape[0]+1))

    # compute the rbf kernel
    Kxy = _np.exp( -pairwise_dists / h**2 / 2)

    dxkxy = -_np.matmul(Kxy, theta)
    sumkxy = _np.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:,i] + _np.multiply(theta[:,i],sumkxy)
    dxkxy = dxkxy / (h**2)
    return (Kxy, dxkxy)


def update(x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
    # Check input
    if x0 is None or lnprob is None:
        raise ValueError('x0 or lnprob cannot be None!')

    theta = _np.copy(x0) 

    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = 0
    for iter in range(n_iter):
        if debug and (iter+1) % 1000 == 0:
            print('iter ' + str(iter+1)) 

        lnpgrad = lnprob(theta)
        # calculating the kernel matrix
        kxy, dxkxy = _svgd_kernel(theta, h = -1)  
        grad_theta = (_np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]  

        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = _np.divide(grad_theta, fudge_factor+_np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad 

    return theta



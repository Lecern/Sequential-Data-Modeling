import numpy as np

def update_mean(X):
    """
    >>> X = np.array([[2, 5], [1, 1], [3, 0], [4, 2], [6, 2]])
    >>> res = [3.2, 2.]
    >>> np.all(update_mean(X) == res)
    True
    """

    return X.sum(axis=0) / X.shape[0]

def calculate_newx(X, u):
    return X - u[np.newaxis, :]

def update_sigma(W, Z):
    W_T = np.c_[W]
    I = np.eye(1)

    res = W.dot(np.linalg.inv(Z)).dot(W_T) + I
    return np.linalg.inv(res)[0,0]

def calculate_z(W, Z, sigma, n_x):
    pre = sigma * W.dot(np.linalg.inv(Z))
    s = pre.dot(n_x.T)
    print('z', s)
    sq = sigma[np.newaxis] + s * s

    return pre.dot(n_x.T), sq

def calculate_suf(n_x, z_l, z_s):
    suf1 = n_x.shape[0]
    dim = n_x.shape[1]
    suf2 = np.zeros((dim, dim))
    for t in n_x:
        suf2 += (np.c_[t]).dot(np.c_[t].T)
    suf3 = np.sum(z_s)
    suf4 = np.zeros((dim, ))
    for t, s in zip(n_x, z_l):
        suf4 += t * s
    
    return suf1, np.diag(np.diag(suf2)), suf3, suf4

def update_parameter(suf1, suf2, suf3, suf4):
    _W = suf4 / suf3
    _Z = suf2 - np.c_[suf4].dot(np.c_[_W].T)
    _Z = np.diag(np.diag(_Z)) / suf1

    return _W, _Z

def calculate_covariance(W, Z):
    print(np.c_[W].dot(np.c_[W].T))
    print(Z)
    return (np.c_[W].dot(np.c_[W].T) + Z)

    

X = np.array([[2, 5], [1, 1], [3, 0], [4, 2], [6, 4]])
u = update_mean(X)
n_x = calculate_newx(X, u)
print('problem 1:', u, n_x)

W = np.array([1, 0])
Z = np.array([[1, 0], [0, 1]])
sigma = update_sigma(W, Z)

z_l, z_s = calculate_z(W, Z, sigma, n_x)
print('problem 2:', sigma, z_l, z_s)

suf1, suf2, suf3, suf4 = calculate_suf(n_x, z_l, z_s)
print(suf1, suf2, suf3, suf4)
W, Z = update_parameter(suf1, suf2, suf3, suf4)
print(W, Z)
print(calculate_covariance(W, Z))
import numpy as np
from scipy.stats import norm

def smoothing(mu, P, V):
    res_mu = []
    res_V = []
    res_mu.append(mu[0])
    res_V.append(V[0])

    for i in range(1, len(mu)):
        J = V[i] / P[i-1]
        print("J", J)
        res_mu.append(mu[i] + J * (res_mu[-1] - mu[i]))
        res_V.append(V[i] + J * (res_V[-1] - P[i-1]) * J)
    
    res_mu = res_mu[::-1]
    res_V = res_V[::-1]

    return res_mu, res_V


def update_parameter(x, sigma, eta, mu, P, log_likelihood):
    if log_likelihood is None:
        _mu_prev = mu
        _P_prev = P
        _log_likelihood = np.log(norm.pdf(x, _mu_prev, np.sqrt(_P_prev + sigma)))
        _x = 0
    else:
        _mu_prev = mu
        _P_prev = P + eta
        _x = np.log(norm.pdf(x, _mu_prev, np.sqrt(_P_prev + sigma)))
        _log_likelihood =  _x + log_likelihood


    _K = _P_prev / (_P_prev + sigma)
    _mu = _mu_prev + _K * (x - _mu_prev)
    _P = (1 - _K) * _P_prev

    return _mu_prev, _P_prev, _log_likelihood, _x, _K, _mu, _P



STUDENT_ID = input("please, input a your id:")
print("Your ID is {}".format(STUDENT_ID))

b_1, b_2, b_3, b_4 = [int(sid) for sid in STUDENT_ID[-4:]]
print("b_1 = {}, b_2 = {}, b_3 = {}, b_4 = {}".format(b_1, b_2, b_3, b_4))

x_1, x_2, x_3, x_4 = map(lambda x: x * 20 + 10, [b_1, b_2, b_3, b_4])
print("x_1 = {}, x_2 = {}, x_3 = {}, x_4 = {}".format(x_1, x_2, x_3, x_4))

INPUT_X = np.vstack((x_1, x_2, x_3, x_4))
sigma = 20
eta = 10
mu = 100
P = 50

mu_list = []
P_list = []
V_list = []

mu_prev, P_prev, log_likelihood, x, K, mu, P = update_parameter(INPUT_X[0], sigma, eta, mu, P, None)
mu_list.append(mu)
P_list.append(P_prev)
V_list.append(P)
print("problem 1:")
print("log_likelihood = {}\nK = {}\nmu = {}\nP = {}\n".format(log_likelihood, K, mu, P))
mu_prev, P_prev, log_likelihood, x, K, mu, P = update_parameter(INPUT_X[1], sigma, eta, mu, P, log_likelihood)
mu_list.append(mu)
P_list.append(P_prev)
V_list.append(P)
print("problem 2:")
print("mu_p = {}\nP_prev = {}\nlog_likelihood = {}\nmarginal = {}\nK = {}\nmu = {}\nP = {}\n".format(mu_prev, P_prev, log_likelihood, x, K, mu, P))
mu_prev, P_prev, log_likelihood, x, K, mu, P = update_parameter(INPUT_X[2], sigma, eta, mu, P, log_likelihood)
mu_list.append(mu)
P_list.append(P_prev)
V_list.append(P)
print("problem 3:")
print("mu_p = {}\nP_prev = {}\nlog_likelihood = {}\nmarginal = {}\nK = {}\nmu = {}\nP = {}\n".format(mu_prev, P_prev, log_likelihood, x, K, mu, P))
mu_prev, P_prev, log_likelihood, x, K, mu, P = update_parameter(INPUT_X[3], sigma, eta, mu, P, log_likelihood)
mu_list.append(mu)
P_list.append(P_prev)
V_list.append(P)
print("problem 4:")
print("mu_p = {}\nP_prev = {}\nlog_likelihood = {}\nmarginal = {}\nK = {}\nmu = {}\nP = {}\n".format(mu_prev, P_prev, log_likelihood, x, K, mu, P))

mu_list = mu_list[::-1]
P_list = P_list[::-1]
V_list = V_list[::-1]

print("mu", mu_list)
print('P', P_list)
print('V', V_list)
print(smoothing(mu_list, P_list, V_list))

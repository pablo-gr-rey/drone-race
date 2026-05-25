from math import log, sqrt, comb
import scipy.optimize


def f(N: int, k: int, beta: float, eps: float) -> float:
    return sum(comb(N, i) * eps**i * (1 - eps) ** (N - i) for i in range(k)) - beta


def find_eps_approx(N: int, k: int, beta: float) -> float:
    return (k - log(beta) + sqrt(log(beta) ** 2 - 2 * k * log(beta))) / N


def find_eps_exact(N: int, k: int, beta: float) -> float:
    return scipy.optimize.brentq(lambda eps: f(N, k, beta, eps), 0, 1)


def ratio(N, k, beta):
    return find_eps_approx(int(N), k, beta) / find_eps_exact(int(N), k, beta)

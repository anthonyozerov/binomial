import os
import random
import numpy as np
from scipy.stats import betabinom
from itertools import product
from pdg_data import load_pdg_data

SETTINGS = [
    "fe",
    "fe-corr",
    "re",
    "birge",
    "re-outliers",
    "offset",
]

RHO_SETTINGS = ["fe-corr"]

TAU_SETTINGS = [
    "re",
    "birge",
    "re-outliers",
    "offset",
]

NOISE_DISTS = [
    "unif",
    "exp",
    "pdg",
]

NS = [3, 4, 6, 8, 10, 12, 15]
TAUS = np.round(np.logspace(-1, 1, 5), 2)
RHOS = [0, 0.1, 0.2]
T = 16000


def simulate(setting, noise_dist, n, tau, rho, empirical_sigmas=None):
    # generate sigma_i's
    if noise_dist == "unif":
        sigmas = np.random.uniform(0.8, 1.2, n)
    elif noise_dist == "exp":
        sigmas = np.random.exponential(1, n)
    elif noise_dist == "pdg":
        assert empirical_sigmas is not None
        sigmas = random.choice(empirical_sigmas)
        sigmas = np.random.choice(sigmas, size=n, replace=False)
        sigmas /= np.median(sigmas)
    if setting == "fe" or (setting == "fe-corr" and rho == 0):
        values = np.random.normal(0, sigmas, n)
    elif setting == "fe-corr":
        values = np.abs(np.random.normal(0, sigmas, n))
        alpha = (1 / rho - 1) / 2
        beta = alpha
        n_neg = betabinom.rvs(n, alpha, beta)
        neg_idx = np.random.choice(np.arange(n), n_neg, replace=False)
        values[neg_idx] = -values[neg_idx]
    elif setting == "re":
        systematic_errors = np.random.normal(0, tau, n)
        random_errors = np.random.normal(0, sigmas, n)
        values = systematic_errors + random_errors
    elif setting == "birge":
        random_errors = np.random.normal(0, sigmas, n)
        values = random_errors * tau
    elif setting == "re-outliers":
        systematic_errors = np.random.standard_cauchy(n) * tau
        random_errors = np.random.normal(0, sigmas, n)
        values = systematic_errors + random_errors
    elif setting == "offset":
        random_errors = np.random.normal(0, sigmas, n)
        values = tau + random_errors
    # elif setting == "re_corr":
    #     if n > 100:
    #         raise ValueError('n too high for re_corr')
    #     Sigma = tau**2 * (0.8 * np.eye(n) + 0.2 * np.ones((n, n)))
    #     rng = np.random.default_rng()
    #     systematic_errors = rng.multivariate_normal(
    #         np.zeros(n), Sigma, method="cholesky"
    #     )
    #     random_errors = np.random.normal(0, sigmas, n)
    #     values = systematic_errors + random_errors
    else:
        raise ValueError("setting not recognized")

    return values, sigmas


def simulate_save(T, setting, noise_dist, n, tau, rho, empirical_sigmas):
    filename = f"{setting}_{noise_dist}_n={n}"
    if tau is not None:
        filename += f"_tau={tau}"
    if rho is not None:
        filename += f"_rho={rho}"
    filename += ".npz"
    filepath = f"data/simulation/{filename}"

    if not os.path.exists(filepath):
        values = np.full((T, n), np.nan)
        sigmas = np.full((T, n), np.nan)

        for t in range(T):
            values[t, :], sigmas[t, :] = simulate(setting, noise_dist, n, tau, rho, empirical_sigmas)
        np.savez(filepath, values=values, sigmas=sigmas)


if __name__ == "__main__":
    for setting in SETTINGS:
        for noise_dist in NOISE_DISTS:
            if noise_dist == 'pdg':
                _, pdg2025_both_dfs, _, _ = load_pdg_data()
            for n, tau, rho in product(NS, TAUS, RHOS):
                if setting not in TAU_SETTINGS:
                    tau = None
                if setting not in RHO_SETTINGS:
                    rho = None
                if noise_dist == 'pdg':
                    empirical_sigmas = [list(df.uncertainty) for df in pdg2025_both_dfs if len(df) >= n]
                else:
                    empirical_sigmas = None
                simulate_save(T, setting, noise_dist, n, tau, rho, empirical_sigmas)

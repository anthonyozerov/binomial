import numpy as np
from scipy.stats import binom, norm, t
import Rmath4
from itertools import product


def I2(values, sigmas):
    """
    Calculate the I^2 statistic for heterogeneity.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates

    Returns:
    --------
    I2 : float
        The I^2 statistic for heterogeneity
    """
    w = 1 / sigmas**2
    ybar = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - ybar) ** 2)
    n = len(values)
    df = n - 1
    I2 = 1 - df / Q
    return np.max([0, I2])


def errscale_test(values, sigmas):
    """
    Test of the Birge ratio model.
    Baker and Jackson (2013)
    """
    sigmas2 = sigmas**2
    w = 1 / sigmas2
    muhat = np.sum(w * values) / np.sum(w)
    Q = np.sum(w * (values - muhat) ** 2)
    n = len(values)

    bi = np.log(sigmas2)
    bbar = np.mean(bi)

    S = np.sum((bi - bbar) * ((values - muhat) ** 2 / sigmas2) / (Q / (n - 1)))

    Sprime = np.zeros(1000)
    for i in range(len(Sprime)):
        eps = np.random.normal(loc=0, scale=1, size=n)
        M = np.sum(1 / sigmas2)
        epshat = 1 / M * np.sum(eps / sigmas)
        Sprime[i] = (
            (n - 1)
            * np.sum((bi - bbar) * (eps - epshat / sigmas) ** 2)
            / np.sum((eps - epshat / sigmas) ** 2)
        )

    p = np.mean(Sprime <= S)
    return p


def fixed_effect(values, sigmas, coverage=None, zalpha=None):
    if zalpha is None:
        # calculate critical value based on desired coverage
        tail_prob = (1 - coverage) / 2
        zalpha = np.abs(norm.ppf(tail_prob))

    w = 1 / sigmas**2
    yhat = np.sum(w * values) / np.sum(w)
    sigma = np.sqrt(1 / np.sum(w))
    interval = [yhat - zalpha * sigma, yhat + zalpha * sigma]
    return interval, yhat, sigma


def random_effects_dl_base(values, sigmas):
    """
    Calculate the base fixed-effects weighted mean, random-effects weights,
    and between-study variance tau^2 (MoM) under the random effects model.

    Used in both the DL and HKSJ methods.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates

    Returns:
    --------
    muhat : float
        The weighted mean estimate under random effects
    wstar : array-like
        The random effects weights for each study
    tau2 : float
        The estimated between-study variance
    """
    # fixed effect weights
    w = 1 / sigmas**2
    # fixed effect weighted mean
    ybar = np.sum(w * values) / np.sum(w)
    # calculate Q statistic (heterogeneity measure)
    Q = np.sum(w * (values - ybar) ** 2)
    n = len(values)
    df = n - 1

    # calculate sums needed for tau^2 estimation
    s1 = np.sum(w)
    s2 = np.sum(w**2)

    # estimate between-study variance (tau^2) using method of moments
    # max with 0 ensures non-negative variance
    tau2 = np.max([0, (Q - df) / (s1 - s2 / s1)])

    # random effects weights
    wstar = 1 / (sigmas**2 + tau2)
    # random effects weighted mean
    muhat = np.sum(values * wstar) / np.sum(wstar)

    return muhat, wstar, tau2


def random_effects_dl(values, sigmas, coverage=None, zalpha=None):
    """
    Compute confidence intervals using DerSimonian and Laird method with the method of moments estimator.

    This implements a random effects model for meta-analysis using the DerSimonian and Laird
    approach with the method of moments estimator for the between-study variance.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates
    coverage : float, optional
        The desired coverage probability of the interval (e.g., 0.95)
    zalpha : float, optional
        The critical value corresponding to the desired coverage

    Returns:
    --------
    interval : list
        The confidence interval [lower, upper]
    muhat : float
        The weighted mean estimate under random effects
    sigma : float
        The standard error of the weighted mean
    tau : float
        The square root of the estimated between-study variance
    """
    assert (
        coverage is not None or zalpha is not None
    ), "Either coverage or zalpha must be provided"

    if zalpha is None:
        # calculate critical value based on desired coverage
        tail_prob = (1 - coverage) / 2
        zalpha = np.abs(norm.ppf(tail_prob))

    muhat, wstar, tau2 = random_effects_dl_base(values, sigmas)

    # standard error of the weighted mean
    sigma = np.sqrt(1 / (np.sum(wstar)))

    # construct confidence interval using normal distribution
    interval = [muhat - zalpha * sigma, muhat + zalpha * sigma]

    return interval, muhat, sigma, np.sqrt(tau2)


def random_effects_hksj(values, sigmas, coverage=None, talpha=None, trunc="none"):
    """
    Compute confidence intervals using the Hartung-Knapp-Sidik-Jonkman method.

    This implements a random effects model for meta-analysis using the HKSJ approach,
    which modifies the variance estimator and uses a t-distribution for inference
    to better account for uncertainty in the estimation of heterogeneity.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates
    coverage : float, optional
        The desired coverage probability of the interval (e.g., 0.95)
    talpha : float, optional
        The critical value from t-distribution corresponding to the desired coverage
    trunc : boolean, optional
        Whether to truncate the scaler above 1, see doi:10.1186/s12874-015-0091-1

    Returns:
    --------
    interval : list
        The confidence interval [lower, upper]
    muhat : float
        The weighted mean estimate under random effects
    sigma : float
        The modified standard error of the weighted mean
    tau : float
        The square root of the estimated between-study variance
    """
    assert coverage is not None or talpha is not None

    n = len(values)
    df = n - 1
    if talpha is None:
        # calculate critical value from t-distribution
        tail_prob = (1 - coverage) / 2
        talpha = np.abs(t.ppf(tail_prob, df))

    muhat, wstar, tau2 = random_effects_dl_base(values, sigmas)

    # modified variance estimator specific to HKSJ method
    Qstar = np.sum(wstar * (values - muhat) ** 2)
    c2 = Qstar / df
    if trunc == "simple":
        c2 = np.max([c2, 1])
    elif trunc == "talpha":
        zalpha = np.abs(norm.ppf(tail_prob))
        c2 = np.max([c2, (zalpha / talpha) ** 2])
    sigma2 = c2 / np.sum(wstar)
    # sigma2 = np.sum(wstar * (values - muhat) ** 2) / (np.sum(wstar) * df)
    sigma = np.sqrt(sigma2)

    # construct confidence interval using t-distribution
    interval = [muhat - talpha * sigma, muhat + talpha * sigma]

    return interval, muhat, sigma, np.sqrt(tau2)


# based on "Combining Information: Statistical Issues and Opportunities for Research"
# (1992) pg 145
# https://nap.nationalacademies.org/catalog/20865/combining-information-statistical-issues-and-opportunities-for-research
def random_effects_mle(values, sigmas, coverage=None, zalpha=None, truth=None):
    """
    Compute confidence intervals using the maximum likelihood estimator.

    This implements a random effects model for meta-analysis using the maximum likelihood estimator.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates
    coverage : float, optional
        The desired coverage probability of the interval (e.g., 0.95)
    zalpha : float, optional
        The critical value corresponding to the desired coverage

    Returns:
    --------
    interval : list
        The confidence interval [lower, upper]
    """
    assert (
        coverage is not None or zalpha is not None
    ), "Either coverage or zalpha must be provided"

    if zalpha is None:
        # calculate critical value based on desired coverage
        tail_prob = (1 - coverage) / 2
        zalpha = np.abs(norm.ppf(tail_prob))

    sigmas2 = sigmas**2
    tau2 = np.var(values)

    for i in range(50):
        w = 1 / (sigmas2 + tau2)
        muhat = np.sum(w * values) / np.sum(w) if truth is None else truth
        tau2 = np.sum(w**2 * ((values - muhat) ** 2 - sigmas2)) / np.sum(w**2)
        tau2 = np.max([0, tau2])

    sigma = np.sqrt(1 / np.sum(w))
    interval = [muhat - zalpha * sigma, muhat + zalpha * sigma]

    return interval, muhat, sigma, np.sqrt(tau2)


def random_effects_pm(values, sigmas, coverage=None):
    """Paule and Mandel"""
    assert coverage is not None, "coverage must be provided"

    n = len(values)
    sigmas2 = sigmas**2
    tau2 = 0
    for i in range(100):
        w = 1 / (sigmas2 + tau2)
        muhat = np.sum(w * values) / np.sum(w)
        Ftau2 = np.sum(w * (values - muhat) ** 2) - (n - 1)
        if tau2 == 0 and Ftau2 < 0:
            break
        if np.isclose(Ftau2, 0):
            break
        delta = Ftau2 / np.sum(w**2 * (values - muhat) ** 2)
        tau2 += delta
    if tau2 != 0:
        assert np.isclose(Ftau2, 0)
    assert tau2 >= 0
    thetahat = np.sum(w * values) / np.sum(w)
    sigma = np.sqrt(1 / np.sum(w))

    zalpha = np.abs(norm.ppf((1 - coverage) / 2))
    interval = [thetahat - zalpha * sigma, thetahat + zalpha * sigma]

    return interval, thetahat, sigma, np.sqrt(tau2)


def birge(
    values,
    sigmas,
    coverage=None,
    dist="normal",
    pdg=False,
    codata=False,
    mle=False,
    truth=None,
):
    """
    Compute confidence intervals using Birge ratio model.

    This method scales the variance by the Birge ratio to account for over-dispersion.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    sigmas : array-like
        The standard errors of the effect estimates
    coverage : float, optional
        The desired coverage probability of the interval (e.g., 0.95)
    zalpha : float, optional
        The critical value corresponding to the desired coverage

    Returns:
    --------
    interval : list
        The confidence interval [lower, upper]
    wm : float
        The weighted mean estimate
    sigma : float
        The standard error of the weighted mean
    chat : float
        The Birge ratio (inflation factor)
    """

    assert (codata is False) or (pdg is False), "codata and pdg cannot be both True"

    n = len(values)

    # calculate precision-weighted standard error
    u = np.sqrt(1 / np.sum(1 / sigmas**2))
    # calculate precision-weighted mean
    if truth is None:
        wm = u**2 * np.sum(values / sigmas**2)
    else:
        wm = truth

    # calculate chi-squared statistic
    if not codata:
        if pdg:
            # use the heuristic of only including studies with small residuals
            # in calculation of chat
            thresh = 3 * np.sqrt(n) * u
            good = sigmas < thresh
            if np.sum(good) <= 1:
                good = np.ones(n, dtype=bool)
        else:
            good = np.ones(n, dtype=bool)
        chi2 = np.sum((values[good] - wm) ** 2 / sigmas[good] ** 2)
        # calculate Birge ratio
        if not mle:
            chat2 = chi2 / (np.sum(good) - 1)
        else:
            chat2 = chi2 / np.sum(good)
        chat = np.sqrt(chat2)
        # ensure the inflation factor is at least 1
        chat = np.max((1, chat))
    else:
        # use the CODATA heuristic of setting chat to the smallest scaling
        # factor such that the maximum standardized residual is <2
        resids_norm = (values - wm) / sigmas
        max_resid = np.max(np.abs(resids_norm))
        chat = max(max_resid / 2, 1)

    # inflate standard error by Birge ratio
    sigma = chat * u

    # calculate critical value
    tail_prob = (1 - coverage) / 2
    zalpha = np.abs(norm.ppf(tail_prob))
    talpha = np.abs(t.ppf(tail_prob, n - 1))
    if dist == "normal":
        crit = zalpha
    elif dist == "t":
        crit = zalpha if chat == 1 else talpha

    else:
        raise ValueError(f"Invalid distribution: {dist}")

    # construct confidence interval
    interval = [wm - crit * sigma, wm + crit * sigma]

    return interval, wm, sigma, chat


def vniim(values, sigmas, coverage=None, zalpha=None, tol=1e-5, max_iter=100):
    assert (
        coverage is not None or zalpha is not None
    ), "Either coverage or zalpha must be provided"

    if zalpha is None:
        # calculate critical value if not provided
        tail_prob = (1 - coverage) / 2
        zalpha = np.abs(norm.ppf(tail_prob))

    # following Taylor 1982
    n = len(sigmas)
    sigmas2 = sigmas**2
    F = n - 1
    diff = 1
    _, wm, _, rb = birge(values, sigmas, zalpha=1)
    ri2 = np.full(sigmas.shape, rb**2)
    resids2 = (wm - values) ** 2 / (sigmas2 * ri2)
    iter = 0
    while diff > tol and iter < max_iter:
        rhs = np.zeros(sigmas.shape)
        for i in range(n):
            # rhs[i] = ri2[i] * (resids2[i]/F) * np.sum(ri2 * (ri2-1))
            rhs[i] = (resids2[i] / F) * np.sum(ri2 * (ri2 - 1))
        ## LHS is ri^4(ri^2-1) = ... (eq 5)
        # LHS is ri^2(ri^2-1) = ... (divide both sides of eq5 by 2)
        # ri^4 - ri^2 - RHS = 0
        # ri^2 = (1 +- sqrt(1 + 4*RHS)) / 2
        # take the +, as other root will be negative
        ri2 = (1 + np.sqrt(1 + 4 * rhs)) / 2
        w = 1 / (sigmas2 * ri2)
        wm = np.sum(values * w) / np.sum(w)
        resids2 = (wm - values) ** 2 / (sigmas2 * ri2)
        chi2 = np.sum(resids2)
        diff = np.abs(chi2 - F)
        iter += 1
    # print(iter)
    u = np.sqrt(1 / np.sum(1 / (sigmas2 * ri2)))
    # u_orig = np.sqrt(1 / np.sum(1 / (sigmas2)))
    # print(np.sqrt(ri2))
    interval = [wm - zalpha * u, wm + zalpha * u]

    return interval, wm


def binomial_method(values, p=0.5, coverage=0.6827, cdf=None, shrink=None):
    """
    Compute confidence intervals using order statistics and the binomial distribution.

    This non-parametric method selects appropriate order statistics to form
    confidence intervals with desired coverage.

    Parameters:
    -----------
    values : array-like
        The effect estimates from individual studies
    p : float, default=0.5
        The probability parameter for the binomial distribution
    coverage : float, default=0.6827
        The target coverage probability (default corresponds to 1-sigma coverage)
    which : str, default="lower"
        Specifies whether to compute the lower or upper bound
    cdf : array-like, optional
        Pre-computed binomial CDF values

    Returns:
    --------
    bound : float
        The selected boundary value
    prob : float
        The actual probability achieved for the bound
    """
    tail_prob = (1 - coverage) / 2

    n = len(values)
    if cdf is None:
        # compute binomial CDF if not provided
        cdf = binom.cdf(np.arange(n + 1), n, p)
    assert len(cdf) == n + 1

    # check that values are sorted (O(n))
    # if not sorted, sort them (O(n log n))
    if not np.all(values[:-1] <= values[1:]):
        values = np.sort(values)

    idx_l = np.argmax(cdf > tail_prob) - 1
    tail_l = cdf[idx_l]
    idx_u = np.argmax(cdf >= 1 - tail_prob)
    tail_u = 1 - cdf[idx_u]
    assert idx_l <= idx_u, f"idx_l={idx_l} > idx_u={idx_u}"

    nominal_coverage = 1 - tail_l - tail_u
    assert (
        nominal_coverage >= coverage
    ), f"Nominal coverage {nominal_coverage} is less than target coverage {coverage}"

    bottom = values[idx_l]
    top = values[idx_u]

    interval = [bottom, top]

    if shrink is None:
        return interval, nominal_coverage, None
    if nominal_coverage == coverage:
        return interval, nominal_coverage, interval

    z_nominal_l = -norm.ppf(tail_l)
    z_nominal_u = -norm.ppf(tail_u)
    z_target = -norm.ppf(tail_prob)

    bottom2 = values[idx_l + 1]
    top2 = values[idx_u - 1]
    tail_l2 = cdf[idx_l + 1]
    tail_u2 = 1 - cdf[idx_u - 1]
    z_nominal_l2 = -norm.ppf(tail_l2)
    z_nominal_u2 = -norm.ppf(tail_u2)
    assert (
        z_nominal_l2 < z_nominal_l
    ), f"z_nominal_l2={z_nominal_l2} < z_nominal_l={z_nominal_l}"
    assert (
        z_nominal_u2 < z_nominal_u
    ), f"z_nominal_u2={z_nominal_u2} < z_nominal_u={z_nominal_u}"

    # if idx_l + 1 > idx_u - 1:
    #     raise NotImplementedError("shrinking not implemented for n=4 50% case")

    if shrink == "scale":
        halfwidth = (top - bottom) / 2
        middle = (bottom + top) / 2
        new_width_b = halfwidth * z_nominal_l / z_target
        new_width_u = halfwidth * z_nominal_u / z_target
        bottom_shrink = middle - new_width_b
        top_shrink = middle + new_width_u
    elif shrink == "center":
        median = values[np.argmin(np.abs(cdf - 0.5))]
        bottom_shrink = median - (median - bottom) * z_target / z_nominal_l
        top_shrink = median + (top - median) * z_target / z_nominal_u
    elif shrink == "cdf-interp":
        xspace = np.linspace(bottom, top, 100, endpoint=True)
        # insert original values into xspace
        xspace = np.unique(np.concatenate([values[idx_l : idx_u + 1], xspace]))
        xspace = np.sort(xspace, kind="mergesort")
        # print(xspace)
        cdf_interp = np.interp(xspace, values, cdf[:-1])
        # print(cdf_interp)
        bottom_shrink = xspace[np.argmax(cdf_interp > tail_prob) - 1]
        top_shrink = xspace[np.argmax(cdf_interp >= 1 - tail_prob)]
    elif shrink == "prob-linear":
        bottom_shrink = bottom + (bottom2 - bottom) * (tail_prob - tail_l) / (
            tail_l2 - tail_l
        )
        top_shrink = top - (top - top2) * (tail_prob - tail_u) / (tail_u2 - tail_u)
    elif shrink == "z-linear":
        bottom_shrink = bottom + (bottom2 - bottom) * (z_nominal_l - z_target) / (
            z_nominal_l - z_nominal_l2
        )
        top_shrink = top - (top - top2) * (z_nominal_u - z_target) / (
            z_nominal_u - z_nominal_u2
        )
    else:
        raise ValueError(f"Invalid shrink method: {shrink}")

    interval_shrink = [bottom_shrink, top_shrink]
    assert (
        interval_shrink[0] >= interval[0]
    ), f"interval_shrink={interval_shrink} is wider than interval={interval}"
    assert (
        interval_shrink[1] <= interval[1]
    ), f"interval_shrink={interval_shrink} is wider than interval={interval}"

    return interval, nominal_coverage, interval_shrink

    # if which == "lower":
    #     # find index where CDF exceeds target probability
    #     idx = np.argmax(cdf > tail_prob) - 1
    #     return values[idx], cdf[idx]
    # elif which == "upper":
    #     # find index where CDF reaches or exceeds 1-target
    #     idx = np.argmax(cdf >= 1 - tail_prob)
    #     return values[idx], 1 - cdf[idx]
    # elif which == "both":
    #     idx_l = np.argmax(cdf > tail_prob) - 1
    #     idx_u = np.argmax(cdf >= 1 - tail_prob)
    #     return values[idx_l], values[idx_u], cdf[idx_l], cdf[idx_u]


# adapted from R wilcox.test
def sign_rank_test(values, h0_median=0):
    values = np.array(values) - h0_median
    zeroes = any(values == 0)
    if zeroes:
        values = values[values != 0]
    n = len(values)

    w = np.add.outer(values, values) / 2
    w = np.sort(w[np.tril_indices(w.shape[0], 0)])

    # count of w > 0
    count = np.sum(w > 0)

    p_upper_tail = Rmath4.psignrank(count - 1, n, 0, 0)
    p_lower_tail = Rmath4.psignrank(count, n, 1, 0)

    return min(min(p_upper_tail, p_lower_tail) * 2, 1)


def sign_rank(values, coverage=0.6827, psignrank=None, qsignrank=None):
    n = len(values)
    alpha = 1 - coverage

    # create vector of Walsh averages
    w = np.add.outer(values, values) / 2
    w = np.sort(w[np.tril_indices(w.shape[0], 0)])

    qu = int(Rmath4.qsignrank(alpha / 2, n, 0, 0))

    if qu == 0:
        qu = 1
    achieved_alpha = 2 * Rmath4.psignrank(qu - 1, n, 0, 0)

    if achieved_alpha > alpha:
        # print('coverage too low')
        qu = qu + 1
        new_achieved_alpha = 2 * Rmath4.psignrank(qu - 1, n, 0, 0)
        assert new_achieved_alpha < achieved_alpha
        achieved_alpha = new_achieved_alpha
        # assert achieved_alpha <= alpha, f'{achieved_alpha} >= {alpha}'

    ql = int(n * (n + 1) / 2 - qu)

    # print(achieved, len(w), qu)
    # lower = w[ql+1-1]
    # upper = w[qu-1]
    lower = w[ql + 1 - 1]
    upper = w[qu - 1]
    interval = [lower, upper]
    # print(lower, upper)
    return interval, 1 - achieved_alpha


def binomial_adapt(values, sigmas, p=0.5, coverage=0.6827, cdf=None, which="smallest"):
    assert len(values) == len(sigmas)
    assert which in ["smallest", "consistency", "random"]
    # values should be sorted in ascending order
    # sigmas should be sorted in the same order as values

    values, sigmas = zip(*sorted(zip(values, sigmas)))
    values = np.array(values)
    sigmas = np.array(sigmas)

    n = len(values)

    if cdf is None:
        cdf = binom.cdf(np.arange(n + 1), n, p)

    idx_l = 0
    prob_l = cdf[idx_l]
    idx_u = 1
    prob_u = 1 - cdf[idx_u]

    idx_l = np.arange(n)
    idx_u = np.arange(n)
    prob_l = cdf[idx_l]
    prob_u = 1 - cdf[idx_u]
    # create nxn matrix of probabilities of within interval
    probs = 1 - prob_l[:, None] - prob_u[None, :]

    # first element in each row where the cumulative sum is greater than target coverage
    idx_row = np.argmax(probs >= coverage, axis=1)

    idx_col = n - np.argmax(probs[::-1, :] >= coverage, axis=0) - 1

    selected_row = np.zeros((n, n), dtype=bool)
    selected_col = np.zeros((n, n), dtype=bool)

    selected_row[np.arange(n), idx_row] = True
    selected_col[idx_col, np.arange(n)] = True

    # only keep the intervals which are both first in their row and first
    # in their column, and where coverage is reached
    interval_good = selected_row & selected_col & (probs >= coverage)

    # (nxn) boolean mat to indices of trues (sx2), where s is the number of intervals
    intervals_idx = np.array(interval_good.nonzero()).T
    # calculate the intervals (sx2)
    intervals = values[intervals_idx]

    if which == "smallest":
        interval_lengths = intervals[:, 1] - intervals[:, 0]  # (s,)
        idx = np.argmin(interval_lengths)

    elif which == "consistency":
        # calculate the error-weighted distance of every value from the interval
        # (nxs)
        errors_l = (
            np.abs((values[:, np.newaxis] - intervals[:, 0])) / sigmas[:, np.newaxis]
        )
        errors_u = (
            np.abs((values[:, np.newaxis] - intervals[:, 1])) / sigmas[:, np.newaxis]
        )

        # we won't consider the points within the interval
        # (nxs)
        within = (values[:, np.newaxis] >= intervals[:, 0]) & (
            values[:, np.newaxis] <= intervals[:, 1]
        )
        # calculate the distance of every value from the interval, 0 if within
        # (nxs)
        dists = np.minimum(errors_l, errors_u) * (~within)
        # get the total dist for each interval
        # (s,)
        total_dist = np.sum(dists, axis=0)

        idx = np.argmin(total_dist)
    elif which == "random":
        idx = np.random.choice(len(intervals))

    interval = intervals[idx]
    interval_idx = intervals_idx[idx]
    prob = probs[interval_idx[0], interval_idx[1]]
    assert prob > coverage

    return interval, prob


def binomial_sigmacdf(values, sigmas, p=0.5, coverage=0.6827):
    tail_prob = (1 - coverage) / 2
    # print(tail_prob)

    # if K is binomial, estimate the CDF of the sum of K 1/sigmas

    n = len(values)
    values, sigmas = zip(*sorted(zip(values, sigmas)))
    sigmas = np.array(sigmas)

    T = 200
    K = binom.rvs(n, p, size=T)
    sums = []

    fsigma = lambda x: 1 / x
    fsigma = lambda x: 1 / np.sqrt(x)

    for k in K:
        s = np.sum(fsigma(np.random.choice(sigmas, size=k, replace=False)))
        sums.append(s)
    sums = np.array(sums)

    sum_thresh = np.quantile(sums, tail_prob, method="lower")
    assert np.mean(sum_thresh > sums) < tail_prob
    # print(np.mean(sum_thresh>sums))

    sigmas_cumsum = np.cumsum(fsigma(sigmas))
    sigmas_cumsum_r = np.cumsum(fsigma(sigmas)[::-1])

    idx_l = np.argmax(sigmas_cumsum > sum_thresh) - 1
    if idx_l == -1:
        prob_l = 0
        interval_l = -np.inf
    else:
        prob_l = np.mean(sigmas_cumsum[idx_l] > sums)
        interval_l = values[idx_l]
    assert prob_l < tail_prob, f"{prob_l} too big"

    # print(sums)
    # print(sum_thresh)
    # print(np.mean(sum_thresh>sums))
    # print(idx_l)
    idx_u_r = np.argmax(sigmas_cumsum_r > sum_thresh) - 1

    if idx_u_r == -1:
        prob_u = 0
        interval_u = np.inf
    else:
        prob_u = np.mean(sigmas_cumsum_r[idx_u_r] > sums)
        idx_u = -idx_u_r
        interval_u = values[idx_u]
    assert prob_u < tail_prob, f"{prob_u} too big"

    # print(prob_l, prob_u)

    # TODO: calculate the probability of the interval (it will be >coverage)
    # assert prob_l < tail_prob, f'{prob_l} too big'
    # assert prob_u < tail_prob, f'{prob_u} too big'
    coverage_nom = 1 - prob_l - prob_u
    assert coverage_nom > coverage, f"coverage={coverage_nom} too small"

    return [interval_l, interval_u], 1 - prob_l - prob_u


def flip_test(values, h0=0, tail="both", mode="median", boot=False):
    if tail not in ["both", "lower", "upper"]:
        raise ValueError("tail must be one of 'both', 'lower', or 'upper'")

    values = np.array(values) - h0
    B = 10000
    n = len(values)
    if 2**n <= B and not boot:
        exact = True
        B = 2**n
    else:
        exact = False

    if mode == "median":
        center_func = np.median
    elif mode == "mean":
        center_func = np.mean
    else:
        raise ValueError("mode must be one of 'median', 'mean'")
    T = center_func(values)

    lower_count = 0
    upper_count = 0

    if exact:
        signs = np.array(list(product([-1, 1], repeat=n)))
        assert signs.shape == (B, n)
    else:
        signs = np.random.choice([-1, 1], size=(B, n))
    if boot:
        values = np.random.choice(values, size=(B, n), replace=True)
    signs_centers = center_func(signs * values, axis=1)
    assert len(signs_centers) == B

    lower_count = np.sum(T < signs_centers)
    upper_count = np.sum(T > signs_centers)

    if tail == "both":
        return (min(lower_count, upper_count) / B) * 2
    elif tail == "lower":
        return lower_count / B
    elif tail == "upper":
        return upper_count / B


def flip_interval(values, coverage=0.6827, mode="median", boot=False, max_iter=1000):
    tail_prob = (1 - coverage) / 2
    # first we binary search between min(values) and max(values) for the lower bound
    range = max(values) - min(values)
    l = min(values) - range
    r = max(values) + range
    pl = 1
    iter = 0
    while (r - l > range * 1e-6 or pl > tail_prob) and iter < max_iter:
        m = (l + r) / 2
        pl = flip_test(values, h0=m, mode=mode, tail="lower", boot=boot)
        if pl < tail_prob:
            l = m
        else:
            r = m
        iter += 1
    lower = m
    if iter == max_iter:
        return [np.nan, np.nan], np.nan
    # now we binary search between lower and max(values) for the upper bound
    l = lower
    r = max(values) + range
    pu = 1
    iter = 0
    while (r - l > range * 1e-6 or pu > tail_prob) and iter < max_iter:
        m = (l + r) / 2
        pu = flip_test(values, h0=m, mode=mode, tail="upper", boot=boot)
        if pu < tail_prob:
            r = m
        else:
            l = m
        iter += 1
    upper = m
    if iter == max_iter:
        return [np.nan, np.nan], np.nan
    assert 1 - (pl + pu) >= coverage

    return [lower, upper], 1 - (pl + pu)


def linear_pool(values, sigmas, coverage=0.6827, gridn=10000):
    n = len(values)

    grid = np.linspace(
        min(values) - 2 * max(sigmas), max(values) + 2 * max(sigmas), gridn
    )
    cdfs = norm.cdf(grid[:, np.newaxis], loc=values, scale=sigmas)
    probs = np.mean(np.nan_to_num(cdfs, nan=0.0), axis=1)

    tail_prob = (1 - coverage) / 2
    assert np.any(probs > tail_prob), f"no left tail values > {tail_prob}"
    assert np.any(probs >= 1 - tail_prob), f"no right tail values >= {1 - tail_prob}"
    idx_l = np.argmax(probs > tail_prob) - 1
    idx_u = np.argmax(probs >= 1 - tail_prob)
    assert idx_l <= idx_u, f"{idx_l} <= {idx_u}"

    achieved_coverage = 1 - probs[idx_l] - (1 - probs[idx_u])

    return [grid[idx_l], grid[idx_u]], achieved_coverage


def birge_forecast(values, sigmas, coverage=0.6827, chat=None):
    """
    Method for interpolating between meta-analysis and forecasting.
    Key difference from birge ratio is that, when chat < 1, instead of
    clipping to 1, we assume that the studies are underdispersed due to
    a large irreducible uncertainty on theta.
    """
    n = len(values)

    sigmas2 = sigmas**2
    # calculate precision-weighted standard error
    u = np.sqrt(1 / np.sum(1 / sigmas2))
    # calculate precision-weighted mean
    wm = u**2 * np.sum(values / sigmas2)

    if chat is None:
        # calculate chi-squared statistic
        chi2 = np.sum((values - wm) ** 2 / sigmas2)
        # calculate Birge ratio
        chat2 = chi2 / (n - 1)
        chat = np.sqrt(chat2)

    sigma_m = chat * u

    if chat < 1:
        sigmas_r2 = (1 - chat**2) * sigmas2
        sigma_r2 = np.median(sigmas_r2)
    else:
        sigma_r2 = 0

    sigma = np.sqrt(sigma_m**2 + sigma_r2)

    interval = [wm - sigma, wm + sigma]

    return interval, wm, sigma, chat, np.sqrt(sigma_r2)


def boot(values, sigmas, coverage=0.6827, which="normal"):
    n = len(values)
    B = 100
    w = 1 / sigmas**2
    tail_prob = (1 - coverage) / 2

    ybar = np.sum(values * w) / np.sum(w)
    sigmahat = np.sqrt(1 / np.sum(w))
    ybars = np.full(B, np.nan)
    sigmahats = np.full(B, np.nan)
    for b in range(B):
        sample = np.random.choice(n, size=n, replace=True)
        ybars[b] = np.sum(values[sample] * w[sample]) / np.sum(w[sample])
        sigmahats[b] = np.sqrt(1 / np.sum(w[sample]))
    if which == "normal":
        sigmahat = np.std(ybars)
        zalpha = np.abs(norm.ppf(tail_prob))
        return [ybar - zalpha * sigmahat, ybar + zalpha * sigmahat]
    elif which == "studentized":
        tstar = (ybars - ybar) / sigmahats

        return np.quantile(ybar - tstar * sigmahat, [tail_prob, 1 - tail_prob])


def interval_score(interval, truth, coverage, percent=False):
    alpha = 1 - coverage
    l, u = interval

    width = u - l
    below = (truth < l) * (2 / alpha) * (l - truth)
    above = (u < truth) * (2 / alpha) * (truth - u)

    total = width + below + above
    if percent:
        total /= np.abs(truth)

    return total

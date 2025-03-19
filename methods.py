import numpy as np
from scipy.stats import binom, norm, t


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


def random_effects_hksj(values, sigmas, coverage=None, talpha=None):
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
    sigma2 = np.sum(wstar * (values - muhat) ** 2) / (np.sum(wstar) * df)
    sigma = np.sqrt(sigma2)

    # construct confidence interval using t-distribution
    interval = [muhat - talpha * sigma, muhat + talpha * sigma]

    return interval, muhat, sigma, np.sqrt(tau2)


def birge(values, sigmas, coverage=None, zalpha=None):
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
    assert (
        coverage is not None or zalpha is not None
    ), "Either coverage or zalpha must be provided"

    if zalpha is None:
        # calculate critical value if not provided
        tail_prob = (1 - coverage) / 2
        zalpha = np.abs(norm.ppf(tail_prob))

    # calculate precision-weighted standard error
    u = np.sqrt(1 / np.sum(1 / sigmas**2))
    # calculate precision-weighted mean
    wm = u**2 * np.sum(values / sigmas**2)
    # calculate chi-squared statistic
    chi2 = np.sum((values - wm) ** 2 / sigmas**2)
    # calculate Birge ratio
    chat2 = chi2 / (len(values) - 1)
    chat = np.sqrt(chat2)
    # ensure the inflation factor is at least 1
    chat = np.max((1, chat))
    # inflate standard error by Birge ratio
    sigma = chat * u

    # construct confidence interval
    interval = [wm - zalpha * sigma, wm + zalpha * sigma]

    return interval, wm, sigma, chat


def binomial_method(values, p=0.5, target=0.15865, which="lower", cdf=None):
    """
    Compute confidence intervals using order statistics and the binomial distribution.

    This non-parametric method selects appropriate order statistics to form
    confidence intervals with desired coverage.

    Parameters:
    -----------
    values : array-like
        The sorted effect estimates from individual studies
    p : float, default=0.5
        The probability parameter for the binomial distribution
    target : float, default=0.15865
        The target tail probability (default corresponds to 1-sigma coverage)
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

    assert which in ["lower", "upper"]
    n = len(values)
    if cdf is None:
        # compute binomial CDF if not provided
        cdf = binom.cdf(np.arange(n + 1), n, p)
    assert len(cdf) == n + 1

    if which == "lower":
        # find index where CDF exceeds target probability
        idx = np.argmax(cdf > target) - 1
        return values[idx], cdf[idx]
    elif which == "upper":
        # find index where CDF reaches or exceeds 1-target
        idx = np.argmax(cdf >= 1 - target)
        return values[idx], 1 - cdf[idx]

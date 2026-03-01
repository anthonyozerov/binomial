import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import warnings


def get_chi2(wm, value, error):
    return np.sum((value - wm) ** 2 / error**2)


def weighted_mean(value, error):
    return np.sum(value / error**2) / np.sum(1 / error**2)


def symmetrize_error(diff, error_n, error_p, method="pdg"):
    assert method in ["pdg", "dimidiated"], f"Invalid method: {method}"
    assert len(diff) == len(error_n) == len(error_p)

    diff = np.array(diff)
    error_n = np.array(error_n)
    error_p = np.array(error_p)

    error = np.full(error_n.shape, np.nan)

    error[diff <= -error_n] = error_n[diff <= -error_n]
    error[diff >= error_p] = error_p[diff >= error_p]
    between = ~((diff <= -error_n) | (diff >= error_p))

    if method == "pdg":
        error_between = (2 * error_n * error_p + diff * (error_p - error_n)) / (
            error_n + error_p
        )
    # elif method == 'linear':
    #     error_between = error_n + diff*(error_p-error_n)
    elif method == "dimidiated":
        error_between = np.where(diff > 0, error_p, error_n)
    error[between] = error_between[between]

    return error


def _pdg_weighted_mean(value, error_n, error_p, init=None, tol=0.01, max_iter=100):
    """Iteratively compute the PDG-style weighted mean and the effective errors."""
    error = 2 * (error_n * error_p) / (error_n + error_p)
    chisq_prev = -999
    wms = []
    for iteration in range(max_iter + 1):
        if iteration == 0 and init is not None:
            wm = init
        elif iteration == max_iter:
            # print(value)
            # print(error_n)
            # print(error_p)
            # wms = np.array(wms)
            # print(wms)
            # plt.plot(np.arange(len(wms)), wms)
            # plt.show()
            warnings.warn("pdg weighted mean iterations exceeded")
        else:
            wm = weighted_mean(value, error)
        wms.append(wm)
        chisq = get_chi2(wm, value, error)
        if np.abs(chisq - chisq_prev) < tol:
            break
        chisq_prev = chisq
        diff = wm - value
        error = symmetrize_error(diff, error_n, error_p)
    return wm, error


def birge_ratio(
    value,
    error_n,
    error_p,
    init=None,
    pdg_br=True,
    pdg_err=True,
    coverage=0.6827,
    scale=True,
):
    n = len(value)
    wm, error = _pdg_weighted_mean(value, error_n, error_p, init=init)
    wm_err = np.sqrt(1 / np.sum(1 / error**2))
    good = np.full(value.shape, True, dtype=bool)
    if pdg_br:
        threshold = 3 * np.sqrt(n) * wm_err
        good = error < threshold
    n_good = np.sum(good)
    if n_good <= 1:
        br = 1
    else:
        br = np.sqrt(get_chi2(wm, value[good], error[good]) / (n_good - 1))

    br_int = birge_ratio_conf(br, n_good)

    scale_factor = max(br, 1)

    if pdg_err:
        wm_err_n = np.sqrt(1 / np.sum(1 / error_n**2))
        wm_err_p = np.sqrt(1 / np.sum(1 / error_p**2))
    else:
        weights = 1 / error**2
        wm_err_n = np.sqrt(1 / np.sum(weights) ** 2 * np.sum(error_n**2 * weights**2))
        wm_err_p = np.sqrt(1 / np.sum(weights) ** 2 * np.sum(error_p**2 * weights**2))

    if scale:
        wm_err_n *= scale_factor
        wm_err_p *= scale_factor

    tail_prob = (1 - coverage) / 2
    zalpha = np.abs(norm.ppf(tail_prob))

    return wm, wm_err_n, wm_err_p, br, br_int


def wm_shift_error(value, error_n, error_p):
    """Return the weighted mean and asymmetric errors via per-point shifts.

    This implements the QUICK_SIGNDP=FALSE method from sbrfit.f.
    For each measurement, we shift it by its upper and lower errors separately,
    refit, and accumulate the squared response. The sign of the response
    determines which error bucket (upper or lower) it contributes to.
    """
    value = value.astype(float)
    error_n = error_n.astype(float)
    error_p = error_p.astype(float)

    wm, _ = _pdg_weighted_mean(value, error_n, error_p)

    # Initialize squared error accumulators
    err_sq_upper = 0.0
    err_sq_lower = 0.0

    for idx in range(len(value)):
        # Shift down by lower error
        shifted = value.copy()
        shifted[idx] = value[idx] - error_n[idx]
        wm_shifted, _ = _pdg_weighted_mean(shifted, error_n, error_p, init=wm)
        response = wm_shifted - wm  # response to shift

        # Use sign of response to determine which error bucket
        if response > 0:
            err_sq_upper += response**2
        else:
            err_sq_lower += response**2

        # Shift up by upper error
        shifted = value.copy()
        shifted[idx] = value[idx] + error_p[idx]
        wm_shifted, _ = _pdg_weighted_mean(shifted, error_n, error_p, init=wm)
        response = wm_shifted - wm  # response to shift

        # Use sign of response to determine which error bucket
        if response > 0:
            err_sq_upper += response**2
        else:
            err_sq_lower += response**2

    wm_err_p = np.sqrt(err_sq_upper)
    wm_err_n = np.sqrt(err_sq_lower)

    return wm, wm_err_n, wm_err_p


def wm_shift_error_quick(value, error_n, error_p):
    """Return the weighted mean and asymmetric errors via per-point shifts (quick method).

    This implements the QUICK_SIGNDP=TRUE method from sbrfit.f.
    For each asymmetric measurement, we shift it by the average error in the direction
    of the larger error, do a single fit, and use scaling factors (RPPF/RPMF) based on
    the measurement's error asymmetry to update the parameter errors.
    """
    value = value.astype(float)
    error_n = error_n.astype(float)
    error_p = error_p.astype(float)

    wm, error_sym = _pdg_weighted_mean(value, error_n, error_p)
    wm_err_sym = np.sqrt(1 / np.sum(1 / error_sym**2))

    # Initialize squared errors to symmetric error squared
    err_sq_upper = wm_err_sym**2
    err_sq_lower = wm_err_sym**2

    for idx in range(len(value)):
        # Skip symmetric measurements
        if error_p[idx] == error_n[idx]:
            continue

        # Calculate average error for this measurement
        error_avg = 0.5 * (error_p[idx] + error_n[idx])

        # Determine shift direction: shift toward larger error
        # +1 if error_p > error_n, -1 otherwise
        shift_dir = np.sign(error_p[idx] - error_n[idx])
        shift_amount = error_avg

        # Shift the measurement (Fortran line 1040)
        shifted = value.copy()
        shifted[idx] = value[idx] + (shift_dir * shift_amount)

        # Refit with shifted measurement (Fortran line 1057)
        wm_shifted, _ = _pdg_weighted_mean(shifted, error_n, error_p)

        # Calculate response (DELR in Fortran, line 1087)
        response = wm_shifted - wm  # DELMFAC = 1.0, so no division needed
        # Clamp response to symmetric error (Fortran line 1088: Sign(Min(Abs(Delr), Sdr(Inode)), Delr))
        # This prevents the response from exceeding the symmetric error, which could happen
        # if the fit is unstable or the measurement has an unusually large influence
        response = np.sign(response) * min(abs(response), wm_err_sym)

        # Determine RPPF and RPMF based on shift direction
        # These essentially represent the amount of asymmetric error which is
        # "not accounted for" by the fit, and can be positive or negative.
        # if shift_dir == 1:
        #     rppf = (error_p[idx] / error_avg)**2 - 1.0
        #     rpmf = (error_n[idx] / error_avg)**2 - 1.0
        # elif shift_dir == -1:
        #     # Swapped assignment
        #     rppf = (error_n[idx] / error_avg)**2 - 1.0
        #     rpmf = (error_p[idx] / error_avg)**2 - 1.0
        # else:
        #     raise ValueError(f"Invalid shift direction: {shift_dir}")
        rppf = (max(error_p[idx], error_n[idx]) / error_avg) ** 2 - 1.0
        rpmf = (min(error_p[idx], error_n[idx]) / error_avg) ** 2 - 1.0

        # Update error squares based on sign of response (Fortran lines 1089-1095)
        if response > 0:
            err_sq_upper += response**2 * rppf
            err_sq_lower += response**2 * rpmf
        else:
            err_sq_lower += response**2 * rppf
            err_sq_upper += response**2 * rpmf

    # Take square root, handling negative case (Fortran lines 1156-1162)
    wm_err_p = wm_err_sym if err_sq_upper < 0 else np.sqrt(err_sq_upper)
    wm_err_n = wm_err_sym if err_sq_lower < 0 else np.sqrt(err_sq_lower)

    return wm, wm_err_n, wm_err_p


def birge_ratio_conf(br, n, coverage=0.6827):
    tail_prob = (1 - coverage) / 2

    chi2_obs = br**2 * (n - 1)

    br_interval = np.sqrt(
        chi2_obs / chi2.ppf(np.array([1 - tail_prob, tail_prob]), df=n - 1)
    )

    return br_interval


def bartlett_linear_ll(theta, value, error_n, error_p):
    sigma = 2 * (error_n * error_p) / (error_n + error_p)
    sigma_prime = (error_p - error_n) / (error_n + error_p)
    ll_perpoint = (
        -0.5 * ((theta - value) / (sigma + sigma_prime * (theta - value))) ** 2
    )
    ll = np.sum(ll_perpoint)
    return ll, ll_perpoint


def pdg_ll(theta, value, error_n, error_p):
    diff = theta - value
    sigma = symmetrize_error(diff, error_n, error_p, method="pdg")
    ll = -0.5 * np.sum(((diff) / (sigma)) ** 2)
    return ll


def dimidiated_ll(theta, value, error_n, error_p):
    diff = theta - value
    # error p when diff > 0, error n when diff < 0
    error = np.where(diff > 0, error_p, error_n)
    ll = -0.5 * np.sum(((diff) / (error)) ** 2)
    return ll


# def pdg_ll(theta, value, error_n, error_p):
#     diff = theta-value
#     sigma = symmetrize_error(diff, error_n, error_p, method='pdg')
#     ll = -0.5 * np.sum(((diff)/(sigma))**2) - np.sum(np.log(sigma))
#     return ll


def ll_interval(value, error_n, error_p, ll_func=pdg_ll):
    lb = np.min(value - 2 * error_n)
    ub = np.max(value + 2 * error_p)
    xspace = np.linspace(lb, ub, 10000)
    lls = np.array([ll_func(x, value, error_n, error_p) for x in xspace])
    diffs = np.diff(lls)
    increasing = np.sign(diffs) == 1
    decreasing = np.sign(diffs) == -1

    # check that it is increasing, then decreasing
    first_dec_idx = np.argmax(decreasing)
    if increasing[first_dec_idx:].any():
        warnings.warn("ll has multiple local maxima")

    ll_mle = np.max(lls)
    mle = xspace[np.argmax(lls)]
    in_interval = lls - ll_mle > -0.5
    lower = xspace[np.argmax(in_interval)]
    upper = xspace[-np.argmax(in_interval[::-1])]
    return mle, mle - lower, upper - mle


# continuous binary search for smallest value
def binary_search(func, target, bounds, tol=1e-5, max_iter=100, decreasing=True):
    l, r = bounds
    for i in range(max_iter):
        # if r - l < tol:
        #     break
        m = (l + r) / 2
        if decreasing:
            if func(m) < target:
                l = m
            else:
                r = m
        else:
            if func(m) > target:
                l = m
            else:
                r = m
    return l


def bartlett_linear_mle(value, error_n, error_p):
    sigma = 2 * (error_n * error_p) / (error_n + error_p)
    sigma_prime = (error_p - error_n) / (error_n + error_p)
    theta_mle = np.mean(value)
    for i in range(100):
        w = sigma / (sigma + sigma_prime * (theta_mle - value)) ** 3
        theta_mle = np.sum(value * w) / np.sum(w)
    mle_ll = bartlett_linear_ll(theta_mle, value, error_n, error_p)

    # function to minimize to obtain the upper and lower points where ll diff is 0.5
    def to_minimize(param):
        theta = param
        ll = bartlett_linear_ll(theta, value, error_n, error_p)
        # print(mle_ll, ll)
        # return np.abs((mle_ll-ll) - 0.5)
        return mle_ll - ll

    upper = binary_search(
        to_minimize,
        0.5,
        (theta_mle, np.max(value) + 10 * np.max(error_p)),
        decreasing=True,
    )
    lower = binary_search(
        to_minimize,
        0.5,
        (np.min(value) - 10 * np.max(error_n), theta_mle),
        decreasing=False,
    )

    assert lower != upper
    assert lower < theta_mle and upper > theta_mle

    return theta_mle, theta_mle - lower, upper - theta_mle

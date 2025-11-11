import numpy as np
from scipy.stats import norm, chi2

def get_chi2(wm, value, error):
    return np.sum((value - wm)**2 / error**2)

def weighted_mean(value, error):
    return np.sum(value / error**2) / np.sum(1 / error**2)

def symmetrize_error(diff, error_n, error_p, method='pdg'):
    error = np.full(error_n.shape, np.nan)

    error[diff < -error_n] = error_n[diff < -error_n]
    error[diff > error_p] = error_p[diff > error_p]
    between = ~((diff < -error_n) | (diff > error_p))

    if method == 'pdg':
        error_between = (2*error_n*error_p+diff*(error_p-error_n))/(error_n+error_p)
    # elif method == 'linear':
    #     error_between = error_n + diff*(error_p-error_n)
    error[between] = error_between[between]

    return error

def birge_ratio(value, error_n, error_p, init=None, pdg_br=True, pdg_err=True, coverage=0.6827):
    n = len(value)
    error = 2 * (error_n * error_p) / (error_n + error_p)

    chisq_prev = -999
    wms = []
    for i in range(101):
        if i == 0 and init is not None:
            wm = init
        elif i == 100:
            raise RuntimeWarning('br iterations exceeded')
        else:
            wm = weighted_mean(value, error)
        chisq = get_chi2(wm, value, error)
        # print(chisq - chisq_prev)
        if np.abs(chisq-chisq_prev) < 0.01:
            break
        chisq_prev = chisq
        wms.append(wm)
        diff = wm-value

        error = symmetrize_error(diff, error_n, error_p)

    wm_err = np.sqrt(1/np.sum(1/error**2))
    good = np.full(value.shape, True, dtype=bool)
    if pdg_br:
        threshold = 3 * np.sqrt(n) * wm_err
        good = error < threshold
    n_good = np.sum(good)
    if n_good <= 1:
        br = 1
    else:
        br = np.sqrt(get_chi2(wm, value[good], error[good])/(n_good-1))
    
    br_int = birge_ratio_conf(br, n_good)

    br = max(br, 1)

    if pdg_err:
        wm_err_n = np.sqrt(1/np.sum(1/error_n**2))
        wm_err_p = np.sqrt(1/np.sum(1/error_p**2))
    else:
        weights = 1/error**2
        wm_err_n = np.sqrt(1/np.sum(weights)**2 * np.sum(error_n**2 * weights**2))
        wm_err_p = np.sqrt(1/np.sum(weights)**2 * np.sum(error_p**2 * weights**2))

    wm_err_n *= br
    wm_err_p *= br

    tail_prob = (1-coverage)/2
    zalpha = np.abs(norm.ppf(tail_prob))

    return wm, wm_err_n, wm_err_p, br, br_int

    # 
    # return wms[-1]

def birge_ratio_conf(br, n, coverage=0.6827):
    tail_prob = (1-coverage)/2

    chi2_obs = br**2 * (n-1)

    br_interval = np.sqrt(chi2_obs/chi2.ppf(np.array([1-tail_prob, tail_prob]), df=n-1))

    return br_interval
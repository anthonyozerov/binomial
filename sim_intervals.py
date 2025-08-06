import os
import numpy as np
import json
import scipy

from methods import (
    birge,
    random_effects_hksj,
    binomial_method,
    random_effects_dl,
    vniim,
    sign_rank,
    flip_interval,
    fixed_effect,
    linear_pool,
    birge_forecast,
    boot,
    random_effects_mle,
    random_effects_pm,
    interval_score,
)

METHODS = [
    "binom",
    "binom-corr",
    "signrank",
    "fe",
    "birge",
    "birge-t",
    "birge-mle",
    "pdg",
    "codata",
    "re_hksj",
    "re_mhksj",
    "re_mmhksj",
    "re_dl",
    "re_mle",
    "re_pm",
    # 'vniim',
    # 'boot-normal',
    # 'boot-student',
    # 'birge-forecast'
    # 'flip',
    # 'lp'
]

REDO = []

TARGET_COV = 0.6827
SIM_DIR = "data/simulation/"
RESULT_DIR = "results/simulation/"


if __name__ == "__main__":
    sim_files = os.listdir(SIM_DIR)

    for fname in sim_files:
        sim_id = fname[:-4]
        print("SIMULATION:", sim_id)
        fpath = os.path.join(SIM_DIR, fname)
        data = np.load(fpath)

        all_values = data["values"]
        all_sigmas = data["sigmas"]
        assert all_values.shape == all_sigmas.shape
        assert len(all_values.shape) == 2
        T = all_values.shape[0]
        n = all_values.shape[1]

        result_path = os.path.join(RESULT_DIR, sim_id + ".json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results = json.load(f)
        else:
            results = {}
        for method in METHODS:
            if method in results and method not in REDO:
                print("skipping", method)
                continue
            print(method + "...")
            results[method] = {}
            results[method]["intervals"] = []

            nominal_cov = TARGET_COV
            if method == "binom":
                _, nominal_cov, _ = binomial_method(
                    np.arange(n), p=0.5, coverage=TARGET_COV
                )
            elif method == "binom-corr":
                # for rho=0.1
                beta_binom_cdf = scipy.stats.betabinom.cdf(
                    np.arange(n + 1), n, a=4.5, b=4.5
                )
                binom_corr_target_cov = 0.65 if n == 3 else TARGET_COV
                (
                    _,
                    nominal_cov,
                    _,
                ) = binomial_method(
                    np.arange(n), cdf=beta_binom_cdf, coverage=binom_corr_target_cov
                )
            elif method == "signrank":
                _, nominal_cov = sign_rank(
                    np.random.uniform(0, 1, size=n), coverage=TARGET_COV
                )
            results[method]["nominal_cov"] = nominal_cov

            for t in range(T):
                values = all_values[t]
                sigmas = all_sigmas[t]

                if method == "fe":
                    interval, _, _ = fixed_effect(values, sigmas, coverage=TARGET_COV)
                elif method in ["birge", "birge-t", "birge-mle", "pdg", "codata"]:
                    dist = "t" if method == "birge-t" else "normal"
                    pdg = method == "pdg"
                    codata = method == "codata"
                    mle = method == "birge-mle"
                    interval, _, _, _ = birge(
                        values,
                        sigmas,
                        coverage=TARGET_COV,
                        dist=dist,
                        pdg=pdg,
                        codata=codata,
                        mle=mle,
                    )
                elif method in ["re_hksj", "re_mhksj", "re_mmhksj"]:
                    if method == "re_mhksj":
                        trunc = "simple"
                    elif method == "re_mmhksj":
                        trunc = "talpha"
                    else:
                        trunc = "none"
                    interval, _, _, _ = random_effects_hksj(
                        values, sigmas, coverage=TARGET_COV, trunc=trunc
                    )
                elif method == "re_dl":
                    interval, _, _, _ = random_effects_dl(
                        values, sigmas, coverage=TARGET_COV
                    )
                elif method == "re_pm":
                    interval, _, _, _ = random_effects_pm(
                        values, sigmas, coverage=TARGET_COV
                    )
                elif method == "re_mle":
                    interval, _, _, _ = random_effects_mle(
                        values, sigmas, coverage=TARGET_COV
                    )
                elif method == "binom":
                    interval, _, _ = binomial_method(values, p=0.5, coverage=TARGET_COV)
                elif method == "binom-corr":
                    interval, _, _ = binomial_method(
                        values, cdf=beta_binom_cdf, coverage=binom_corr_target_cov
                    )
                elif method == "signrank":
                    interval, _ = sign_rank(values, coverage=TARGET_COV)
                results[method]["intervals"].append(interval)

                del interval
        for method in METHODS:
            intervals = np.array(results[method]["intervals"])
            # results[method]['intervals'] = intervals

            results[method]["coverage"] = np.mean(
                (intervals[:, 0] <= 0) & (0 <= intervals[:, 1])
            ).tolist()
            results[method]["widths"] = (intervals[:, 1] - intervals[:, 0]).tolist()
        with open(result_path, "w") as f:
            json.dump(results, f)

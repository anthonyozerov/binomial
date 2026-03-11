import os
import json

import numpy as np
from scipy.stats import norm
from itertools import combinations, pairwise
from collections import defaultdict
from tqdm import tqdm
import pdg_methods


def measurement_dist(
    dfs,
    thetas=None,
    sigma_thetas=None,
    weight="quantity",
    lists=False,
    ci=True,
    start_year=None,
    end_year=None,
    n_recent=None,
    asym_error=None,
    asym_error_method="pdg",
    skip=False,
    return_samples=False,
):
    """
    Compute measurement distribution statistics for a sequence of dataframes in `dfs`.

    If each dataframe contains measurements of the same quantity, this evaluates the spread of the
    measurements around each other, around the weighted mean, and around a ground truth. The output
    is a dictionary which decribes the tail of the empirical distributions of the spread statistics.

    Mathematical details:
    - For each dataframe, consider the measurements as y_i, i=1,...,n with uncertainties σ_i.
    - For each quantity: Calculate a weighted mean (ybar) and uncertainty (S) using PDG methods.
    - The spread statistics:
        * z_ij = |y_i - y_j| / sqrt(σ_i² + σ_j²)
        * h_i = |y_i - ybar| / sqrt(σ_i² + 1/S)
          (this has the incorrect normalization from Roos et al)
        * h_i' = |y_i - ybar| / sqrt(σ_i² * (1 - 1/(σ_i² * S))² + (S - 1/σ_i²) / S²)
          (this is the correct normalization)
        * h_i* = |y_i - theta| / sqrt(σ_i² + σ_theta²)
    - Measurement error handling:
        * If symmetric errors, use the `uncertainty` column from `df`.
        * If asymmetric errors, use the specified columns in `asym_error`. For a given comparison
          (e.g. y_i-y_j, or y_i-ybar), the error on y_i is calculated via pdg_methods.symmetrize_error.
    - Weighting:
        * If `weight` is "quantity", each quantity has equal weight; so each z_ij is weighted by 1/n_pairs
          in that quantity, and each h_i is weighted by 1/n in that quantity.
        * If `weight` is "measurement", each measurement has equal weight; so each z_ij is weighted by 1/(n-1)
          in that quantity, and each h_i is weighted by 1 in that quantity.
    - Confidence intervals:
        * If `ci` is True, bootstrap confidence intervals are computed for the empirical tail probability
          at the different nominal 'sig' sigma levels. Bootstrapping is done by sampling quantities with replacement.

    Args:
        dfs (list[pandas.DataFrame]): Each DataFrame should include 'value' and 'year' columns.
        thetas (optional, np.ndarray): True values of the quantities.
        sigma_thetas (optional, np.ndarray): Uncertainties of the true `thetas` values.
        weight (str): Weighting scheme, "quantity" or "measurement".
        lists (bool): If True, return lists instead of numpy arrays (use for json serialization).
        ci (bool): If True, also compute (weighted) bootstrap confidence intervals.
        start_year (int, optional): Use only data after this year (inclusive).
        end_year (int, optional): Use only data before this year (inclusive).
        n_recent (int, optional): Only use the n most recent measurements per dataframe.
        asym_error (tuple[str, str], optional): Column names for asymmetric (-, +) uncertainties.
        asym_error_method (str): How to treat asymmetric errors, passed to pdg_methods.symmetrize_error.

    Returns:
        dict: Collected statistics. Either dict of numpy arrays or dict of lists (if `lists` is True).

    """
    assert weight in ["quantity", "measurement"]

    if thetas is not None and sigma_thetas is None:
        sigma_thetas = np.zeros_like(thetas)

    # indices to the quantities which the zs and pair_weights came from
    # will be used to bootstrap
    quantity_idxs_pair = []
    quantity_idxs = []

    zs = []
    pair_weights = []  # the unnormalized weights which will be applied to the zs
    pair_years = []  # the years of the two studies in the pair
    pair_techniques = []

    hs = []  # Roos et al statistic
    hprimes = []  # corrected Roos et al statistic
    hstars = []  # Differences from ground truth
    weights = []  # the unnormalized weights which will be applied to the hs, hprimes, and hstars

    for quantity_idx, df in enumerate(dfs):
        df_sub = df.copy()

        if start_year is not None:
            df_sub = df_sub[df_sub["year"] >= start_year]
        if end_year is not None:
            df_sub = df_sub[df_sub["year"] <= end_year]

        if n_recent is not None:
            # get the n most recent measurements according to year
            df_sub = df_sub.sort_values(by="year", ascending=False).head(n_recent)

        n = len(df_sub)
        if n <= 1:
            if skip:
                continue
            else:
                raise ValueError(f"n={n} for df {quantity_idx} with length {len(df)}")

        n_pairs = int(n * (n - 1) / 2)
        # print(f"n: {n}, n_pairs: {n_pairs}")
        # weights for pairs
        if weight == "quantity":
            pair_weights += list([1 / n_pairs] * n_pairs)
            weights += list([1 / n] * n)
        elif weight == "measurement":
            pair_weights += list([1 / (n - 1)] * n_pairs)
            weights += list([1] * n)

        # store the index of this quantity
        quantity_idxs += list([quantity_idx] * n)
        quantity_idxs_pair += list([quantity_idx] * n_pairs)

        values = np.array(df_sub["value"])

        cols = df_sub.columns
        years = np.array(df_sub["year"]) if "year" in cols else np.full(n, np.nan)
        techniques = (
            np.array(df_sub["technique"]) if "technique" in cols else np.full(n, np.nan)
        )

        if asym_error is None:
            assert "uncertainty" in df_sub.columns, "uncertainty column not found"
            error_n = np.array(df_sub["uncertainty"])
            error_p = np.array(df_sub["uncertainty"])

        else:
            assert asym_error[0] in df_sub.columns, f"column {asym_error[0]} not found"
            assert asym_error[1] in df_sub.columns, f"column {asym_error[1]} not found"
            error_n = np.array(df_sub[asym_error[0]])
            error_p = np.array(df_sub[asym_error[1]])

        assert (
            len(values) == len(error_n) == len(error_p) == n
        ), f"len(values)={len(values)}, len(error_n)={len(error_n)}, len(error_p)={len(error_p)}, n={n}"
        # calculate h and hprime: the standardized differences from the weighted mean

        # calculate weighted mean
        ybar, _ = pdg_methods._pdg_weighted_mean(values, error_n, error_p)
        # calculate the one-sided error on each measurement
        error = pdg_methods.symmetrize_error(
            values - ybar, error_n, error_p, method=asym_error_method
        )
        # calculate error on weighted mean (note: we treat it as symmetric here)
        sigma2 = error**2
        S = np.sum(1 / sigma2)

        # denominators of h and hprime
        std = np.sqrt(sigma2 + 1 / S)
        stdprime = np.sqrt(
            sigma2 * (1 - 1 / (sigma2 * S)) ** 2 + (S - 1 / sigma2) / (S**2)
        )

        hs += list(np.abs(values - ybar) / std)
        hprimes += list(np.abs(values - ybar) / stdprime)

        # calculate hstar: the difference from a ground truth or reference value
        if thetas is not None:
            theta = thetas[quantity_idx]
            sigma_theta = sigma_thetas[quantity_idx]
            error_theta = np.where(values > theta, error_p, error_n)  # dimidiated error
            stdstar = np.sqrt(error_theta**2 + sigma_theta**2)
            hstars += list(np.abs(values - theta) / stdstar)

        # calculate the z statistic for each pair of measurements
        val_list = [
            {
                "value": v,
                "error_n": error_n,
                "error_p": error_p,
                "year": y,
                "technique": t,
            }
            for v, error_n, error_p, y, t in zip(
                values, error_n, error_p, years, techniques
            )
        ]

        pairs = list(combinations(val_list, 2))
        assert len(pairs) == n_pairs, f"len(pairs) {len(pairs)} != n_pairs {n_pairs}"
        for pair in pairs:
            diff = pair[0]["value"] - pair[1]["value"]

            # calculate one error for each measurement from the asymmetric errors,
            # depending on the value of the difference.
            # Will have no effect if the errors are symmetric.
            errors = pdg_methods.symmetrize_error(
                [-diff, diff],
                # [pair[0]['value'], pair[1]['value']],
                [pair[0]["error_n"], pair[1]["error_n"]],
                [pair[0]["error_p"], pair[1]["error_p"]],
                method=asym_error_method,
            )
            assert len(errors) == 2

            # calculate the z statistic
            z = np.abs(diff) / np.sqrt(np.sum(errors**2))
            zs.append(z)
            earlier = np.argmin([pair[0]["year"], pair[1]["year"]])
            later = (earlier + 1) % 2
            pair_years.append((pair[earlier]["year"], pair[later]["year"]))
            pair_techniques.append(
                (pair[earlier]["technique"], pair[later]["technique"])
            )
    assert len(zs) == len(quantity_idxs_pair) == len(pair_weights)
    assert len(hs) == len(quantity_idxs) == len(weights)
    assert len(hprimes) == len(quantity_idxs) == len(weights)
    if thetas is not None:
        assert len(hstars) == len(quantity_idxs) == len(weights)

    quantity_idxs = np.array(quantity_idxs)
    quantity_idxs_pair = np.array(quantity_idxs_pair)
    quantities = np.unique(quantity_idxs)

    zs = np.array(zs)
    hs = np.array(hs)
    hprimes = np.array(hprimes)
    if thetas is not None:
        hstars = np.array(hstars)
    pair_weights = np.array(pair_weights)
    weights = np.array(weights)

    # set up output dictionary
    output = defaultdict(list)

    # calculate the right-tail probability for each statistic, for each z in zspace
    zspace = np.linspace(0, 10, 1000)
    output["zspace"] = zspace

    # sum of the weights, used to normalize and compute probability
    pair_denom = np.sum(pair_weights)
    denom = np.sum(weights)

    # calculate the tail probabilities for each metric at each z in zspace
    for z in zspace:
        output["pair"].append(np.sum((zs > z) * pair_weights) / pair_denom)
        output["h"].append(np.sum((hs > z) * weights) / denom)
        output["hprime"].append(np.sum((hprimes > z) * weights) / denom)
        if thetas is not None:
            output["hstar"].append(np.sum((hstars > z) * weights) / denom)

    # add a folded standard normal distribution to the output for convenience
    output["norm"] = norm.cdf(-zspace) * 2

    if return_samples:
        output["zs"] = zs
        output["hs"] = hs
        output["hprimes"] = hprimes
        if thetas is not None:
            output["hstars"] = hstars

        output["weights"] = weights
        output["pair_weights"] = pair_weights

        output["quantity_idxs"] = quantity_idxs
        output["quantity_idxs_pair"] = quantity_idxs_pair
        output["pair_years"] = pair_years
        output["pair_techniques"] = pair_techniques

    # This section creates a bootstrap confidence interval for the empirical tail probability
    # at the different 'sig' sigma levels
    if ci:
        sigs = [0.5, 1, 2, 5]
        metrics = [m for m in ["pair", "h", "hprime", "hstar"] if m in output]
        metric_map = {"pair": zs, "h": hs, "hprime": hprimes}
        if "hstar" in metrics:
            metric_map["hstar"] = hstars

        # calculate non-bootstrapped tail probabilities for each metric at each sigma level
        tail_probs = defaultdict(dict)
        for metric in metrics:
            if metric == "pair":
                w = pair_weights / np.sum(pair_weights)
            else:
                w = weights / np.sum(weights)
            for sig in sigs:
                tail_probs[metric][sig] = np.sum((metric_map[metric] > sig) * w)

        # calculate bootstrap samples of the tail probabilities for each metric at each sigma level
        sample_tail_probs = defaultdict(lambda: defaultdict(list))
        np.random.seed(0)
        for b in tqdm(range(200)):
            # sample of quantities
            quantity_idxs_sample = np.random.choice(
                quantities, size=len(quantities), replace=True
            )
            # count how many times each quantity appears
            unique, counts = np.unique(quantity_idxs_sample, return_counts=True)
            # map the quantity indices to the counts
            quantity_idx_to_count = defaultdict(int)
            for unique_idx, count in zip(unique, counts):
                quantity_idx_to_count[unique_idx] = count

            # set weights to original weights multiplied by the count corresponding to the quantity
            # here we are taking advantage of the equivalence between bootstrapping and reweighting
            pair_weights_sample = pair_weights * np.array(
                [quantity_idx_to_count[idx] for idx in quantity_idxs_pair]
            )
            weights_sample = weights * np.array(
                [quantity_idx_to_count[idx] for idx in quantity_idxs]
            )

            # calculate tail probabilities in the bootstrap sample for each metric at each sigma level
            for metric in metrics:
                weights_metric_sample = (
                    pair_weights_sample if metric == "pair" else weights_sample
                )
                norm_weights = weights_metric_sample / np.sum(weights_metric_sample)

                for sig in sigs:
                    sample_tail_probs[metric][sig].append(
                        np.sum((metric_map[metric] > sig) * norm_weights)
                    )

        # calculate bootstrap confidence intervals for the tail probabilities for each metric at each sigma level
        for metric in metrics:
            for sig in sigs:
                quantiles = np.quantile(sample_tail_probs[metric][sig], [0.025, 0.975])
                boot_ci = [
                    2 * tail_probs[metric][sig] - quantiles[1],
                    2 * tail_probs[metric][sig] - quantiles[0],
                ]
                output[metric + "_boot_ci_" + str(sig)] = boot_ci
                output[metric + str(sig)] = [tail_probs[metric][sig]]

    if lists:
        for k, v in output.items():
            output[k] = list(v)
    else:
        for k, v in output.items():
            output[k] = np.array(v)
    return output


ORDER = [
    "historical",
    "bailey-constants",
    "chemical",
    "particle",  # this is PDG 1970
    "pdg2025-stat",
    "pdg2025-both",
    "baker-pdg2011-stat",
    "baker-pdg2011-both",
    "bailey-pdg",
    "bailey-pdg-stable",
    "bipm-radionuclide",
    "bailey-nuclear",
    "bailey-interlab",
    "bailey-interlab-key",
    "manylabs2",
    "baker-medical",
    "bailey-medical",
    "cochrane-dich",
]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # point matplotlib ticks inwards
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    # add top and right ticks
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    # use tex
    plt.rcParams["text.usetex"] = True

    nrow = 5
    ncol = 4

    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(8, 8),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    files = os.listdir("results/measurement_dist")

    legend_ax = axs.flatten()[0]
    for i, file in enumerate(files):
        assert file.split(".")[0] in ORDER, f"{file} not in ORDER"
        position = ORDER.index(file.split(".")[0])

        with open(os.path.join("results/measurement_dist", file), "r") as f:
            output = json.load(f)
        ax = axs.flatten()[position]

        name = output["name"]
        zspace = np.array(output["zspace"])

        probs = np.array(output["pair"])
        ax.plot(zspace[probs > 0], probs[probs > 0], label=r"$z_{ij}$", color="red")
        # probs = np.array(output["h"])
        # ax.plot(zspace[probs > 0], probs[probs > 0], label=r"$h_{i}$", color='lightblue')
        probs = np.array(output["hprime"])
        ax.plot(
            zspace[probs > 0], probs[probs > 0], label=r"$h_{i}^\prime$", color="blue"
        )
        # if "hstar" in output:
        #     probs = np.array(output["hstar"])
        #     ax.plot(
        #         zspace[probs > 0], probs[probs > 0], label=r"$h_{i}^\star$", color="red"
        #     )
        #     legend_ax = ax
        probs = np.array(output["norm"])
        ax.plot(
            zspace, probs, label=r"$|\mathcal{N}(0,1)|$", color="black", linestyle="--"
        )
        # ax.legend()
        ax.text(0.95, 0.95, name, transform=ax.transAxes, ha="right", va="top")
        if position % ncol == 0:
            ax.set_ylabel("$P(Z>z)$")
        ax.set_xlabel("$z$")

        if "pair0.5" in output:
            for sig in [0.5, 1, 2, 5]:
                val = output["pair" + str(sig)][0]
                ci = np.array([output["pair_boot_ci_" + str(sig)]]).T
                ci[0, :] = val - ci[0, :]
                ci[1, :] = ci[1, :] - val
                ax.errorbar([sig], [val], yerr=ci, color="red")

        ax.tick_params(which="both", top=True, right=True)
    empty_axs = axs.flatten()[i + 1 :]
    bottom_axs = axs.flatten()[i + 1 - ncol : i + 1]
    for ax in empty_axs:
        # ax.axis('off')
        ax.set_axis_off()
    for ax in bottom_axs:
        ax.xaxis.set_tick_params(labelbottom=True)
    handles, labels = legend_ax.get_legend_handles_labels()

    axs.flatten()[-1].legend(
        handles=handles, labels=labels, frameon=False, loc="lower right"
    )

    plt.ylim(5e-4, 1)
    plt.xlim(0, 9)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(f"figs/measurement_dist.pdf", bbox_inches="tight")
    plt.savefig(f"figs/measurement_dist.png", bbox_inches="tight", dpi=300)

    plt.xlim(0, 2.5)
    plt.yscale("linear")
    plt.ylim(0, 1)
    plt.savefig(f"figs/measurement_dist_zoom.pdf", bbox_inches="tight")
    plt.savefig(f"figs/measurement_dist_zoom.png", bbox_inches="tight", dpi=300)

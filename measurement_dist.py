import os
import json

import numpy as np
from scipy.stats import norm
from itertools import combinations
from collections import defaultdict


def measurement_dist(
    dfs, thetas=None, sigma_thetas=None, weight="quantity", lists=False
):
    assert weight in ["quantity", "measurement"]
    zs = []
    hs = []
    hprimes = []
    hstars = []
    weights = []
    pair_weights = []

    for j, df in enumerate(dfs):
        n = len(df)
        n_pairs = int(n * (n - 1) / 2)
        # weights for pairs
        if weight == "quantity":
            pair_weights += list([1 / n_pairs] * n_pairs)
            weights += list([1 / n] * n)
        elif weight == "measurement":
            pair_weights += list([1 / (n - 1)] * n_pairs)
            weights += list([1] * n)

        values = np.array(df["value"])
        sigmas = np.array(df["uncertainty"])

        sigma2 = sigmas**2
        S = np.sum(1 / sigma2)

        ybar = np.sum(values / sigma2) / S

        std = np.sqrt(sigma2 + 1 / S)
        stdprime = np.sqrt(
            sigma2 * (1 - 1 / (sigma2 * S)) ** 2 + (S - 1 / sigma2) / (S**2)
        )

        hs += list(np.abs(values - ybar) / std)
        hprimes += list(np.abs(values - ybar) / stdprime)
        if thetas is not None:
            theta = thetas[j]
            sigma_theta = sigma_thetas[j]
            stdstar = np.sqrt(sigma2 + sigma_theta)
            hstars += list(np.abs(values - theta) / stdstar)

        val_list = list(zip(values, sigmas))
        for pair in combinations(val_list, 2):
            z = np.abs(pair[0][0] - pair[1][0]) / np.sqrt(
                pair[0][1] ** 2 + pair[1][1] ** 2
            )
            zs.append(z)

    zs = np.array(zs)
    pair_weights = np.array(pair_weights)
    weights = np.array(weights)

    zspace = np.linspace(0, 10, 1000)
    output = defaultdict(list)
    output["zspace"] = zspace

    pair_denom = np.sum(pair_weights)
    denom = np.sum(weights)
    for z in zspace:
        output["pair"].append(np.sum((zs > z) * pair_weights) / pair_denom)
        output["h"].append(np.sum((hs > z) * weights) / denom)
        output["hprime"].append(np.sum((hprimes > z) * weights) / denom)
        if thetas is not None:
            output["hstar"].append(np.sum((hstars > z) * weights) / denom)

    output["norm"] = norm.cdf(-zspace) * 2
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
        ax.plot(zspace[probs > 0], probs[probs > 0], label=r"$z_{ij}$")
        probs = np.array(output["h"])
        ax.plot(zspace[probs > 0], probs[probs > 0], label=r"$h_{i}$")
        probs = np.array(output["hprime"])
        ax.plot(zspace[probs > 0], probs[probs > 0], label=r"$h_{i}^\prime$")
        if "hstar" in output:
            probs = np.array(output["hstar"])
            ax.plot(
                zspace[probs > 0], probs[probs > 0], label=r"$h_{i}^\star$", color="red"
            )
            legend_ax = ax
        probs = np.array(output["norm"])
        ax.plot(zspace, probs, label=r"$|\mathcal{N}(0,1)|$", color="black")
        # ax.legend()
        ax.text(0.95, 0.95, name, transform=ax.transAxes, ha="right", va="top")
        if position % ncol == 0:
            ax.set_ylabel("$P(Z>z)$")
        ax.set_xlabel("$z$")

        ax.tick_params(which="both", top=True, right=True)
    empty_axs = axs.flatten()[i + 1 :]
    for ax in empty_axs:
        # ax.axis('off')
        ax.set_axis_off()
    handles, labels = legend_ax.get_legend_handles_labels()

    axs.flatten()[-1].legend(
        handles=handles, labels=labels, frameon=False, loc="lower right"
    )

    # fig.legend(handles, labels, loc='lower right')
    plt.ylim(1e-4, 1)
    plt.xlim(0, 10)
    plt.yscale("log")
    # plt
    # plt.legend()
    plt.tight_layout()
    plt.savefig("figs/measurement_dist.pdf", bbox_inches="tight")
    # plt.show()

import os
import re
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
from scipy.stats import norm

from measurement_dist import measurement_dist
from pdg_data import load_pdg_data, pdgid_type_map

plt.rcParams.update(
    {
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
    }
)

os.makedirs("../figs/blinding", exist_ok=True)

# Year when blinding became widespread in particle physics experiments.
# Pairs are split into "before" (both years < BLINDING_YEAR) and
# "after" (both years >= BLINDING_YEAR).
BLINDING_YEAR = 2000
USE_HIDDEN = False
MIN_YEAR = -np.inf
EXCLUDE_BR = False
ONLY_BR = True

PREFIX = "H_" if USE_HIDDEN else ""

ROW_LABELS = [
    r"Average $|z_{ij}|$",
    r"Average $z_{ij}^2$",
    r"Fraction $|z_{ij}| > 1$",
    r"Fraction $|z_{ij}| > 2$",
]
# Under H0 (z_ij ~ N(0,1)): E[|z|] = sqrt(2/pi), E[z^2] = 1,
# P(|z|>1) = 2*Phi(-1), P(|z|>2) = 2*Phi(-2).
EXPECTATIONS = [
    f"{np.sqrt(2/np.pi):.3f}",
    "1",
    f"{2*norm.cdf(-1):.4f}",
    f"{2*norm.cdf(-2):.4f}",
]

# Collaborations classified as blinded or non-blinded based on known practices.
# These lists are used to split pairs into blinded/non-blinded subsets.
BLINDED_COLLABS = [
    "BABR",
    "BELL",
    "D0",
    "ATLS",
    "CMS",
    "LHCB",
    "BES3",
    "CLE3",
    "BEL2",
    "E835",
    "SLD",
]
NOT_BLINDED_COLLABS = [
    "ARG",
    "ALEP",
    "DLPH",
    "L3",
    "OPAL",
    "BES",
    "BES2",
    "CLEO",
    "CLE2",
    "CBAL",
    "CBAR",
    "CMD",
    "CMD2",
    "CMD3",
    "DM1",
    "DM2",
    "E760",
    "LASS",
    "GAM2",
    "GAM4",
    "KEDR",
    "MPS",
    "MPS2",
    "MPSF",
    "MRK1",
    "MRK2",
    "MRK3",
    "OBLX",
    "OMEG",
    "SND",
    "VES",
]


# --- helper functions ---


def parse_measurement(s: str) -> tuple[float, float, float]:
    if "@" in s:
        return (np.nan, np.nan, np.nan)
    s = s.strip()

    # ------------------------------------------------------------------ #
    # 1.  Strip trailing exponent  (last E/e followed by optional sign + digits)
    # ------------------------------------------------------------------ #
    exp_match = re.search(r"[Ee]([+-]?\d+)$", s)
    scale = 1.0
    if exp_match:
        scale = 10 ** int(exp_match.group(1))
        s = s[: exp_match.start()]

    # ------------------------------------------------------------------ #
    # 2.  Tokenise                                                         #
    # ------------------------------------------------------------------ #
    # Each error token is preceded by one or two sign characters.
    # We split on positions where such a sign sequence begins.
    # Regex for one token: (sign-chars)(unsigned number)
    #   sign-chars: one of  +  |  -  |  +-  |  -+   (normalise to ±)
    #   unsigned number: digits with optional decimal point
    token_re = re.compile(r"([+-]{1,2})(\d+(?:\.\d+)?)")

    # Central value: everything before the first token_re match that
    # does NOT start at position 0.
    first = token_re.search(s, pos=1)
    if first:
        value = float(s[: first.start()])
        rest = s[first.start() :]
    else:
        value = float(s)
        rest = ""

    pos_errors: list[float] = []
    neg_errors: list[float] = []

    for signs, mag in token_re.findall(rest):
        x = float(mag)
        if "+" in signs:
            pos_errors.append(x)
        if "-" in signs:
            neg_errors.append(x)

    # ------------------------------------------------------------------ #
    # 3.  Combine in quadrature and apply scale                          #
    # ------------------------------------------------------------------ #
    def quad(errs: list[float]) -> float:
        return np.sqrt(sum(e**2 for e in errs)) if errs else 0.0

    return (value * scale, quad(pos_errors) * scale, quad(neg_errors) * scale)


def quantity_weights(mask, quantity_idxs_pair, boot_count=None):
    """Normalized pair weights (1/n_pairs_per_quantity), optionally reweighted by boot_count."""
    idxs = quantity_idxs_pair[mask]
    # Each pair gets weight 1/n, where n is the number of pairs for its quantity,
    # so every physical quantity contributes equally to weighted statistics.
    _, inv, counts = np.unique(idxs, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    if boot_count is not None:
        # Multiply by bootstrap resample count to implement quantity-level resampling.
        w *= np.array([boot_count[q] for q in idxs])
    return w / w.sum()


def weighted_stats(zs_mask, w):
    """Weighted [avg|z|, avg z^2, frac|z|>1, frac|z|>2]."""
    # The four statistics summarize departure from the N(0,1) null;
    # each is a weighted mean of a function of |z_ij| over pairs.
    return np.array(
        [
            np.dot(w, zs_mask),  # E[|z|],     null = sqrt(2/pi)
            np.dot(w, zs_mask**2),  # E[z^2],     null = 1
            np.dot(w, zs_mask > 1),  # P(|z|>1),   null = 2*Phi(-1) ~ 0.317
            np.dot(w, zs_mask > 2),  # P(|z|>2),   null = 2*Phi(-2) ~ 0.046
        ]
    )


def shared_quantity_mask(mask_a, mask_b, quantity_idxs_pair):
    """Mask of pairs whose quantity appears in both mask_a and mask_b."""
    # Restricts to quantities with pairs in both groups, so before/after
    # comparisons are not confounded by a different mix of quantities.
    shared = np.intersect1d(
        np.unique(quantity_idxs_pair[mask_a]),
        np.unique(quantity_idxs_pair[mask_b]),
    )
    return np.isin(quantity_idxs_pair, shared)


def survival_curve(zs_mask, w, zspace):
    """Weighted survival curve P(|Z| > z) over zspace."""
    # Broadcasting: zs_mask[:,None] > zspace[None,:] gives an (n_pairs, n_z) boolean
    # matrix; left-multiplying by w (shape n_pairs) gives the weighted empirical
    # survival function at each z in zspace.
    return w @ (zs_mask[:, None] > zspace[None, :])


def bootstrap_ci_errors(center, boot_samples, axis=-1):
    """Bootstrap 95% CI as (neg_error, pos_error) using basic bootstrap."""
    # Basic (reflection) bootstrap: if theta_hat is the observed statistic and
    # Q_alpha is the alpha quantile of the bootstrap distribution theta_hat*,
    # the 95% CI is [2*theta_hat - Q_0.975, 2*theta_hat - Q_0.025].
    # This corrects for bias in the bootstrap distribution.
    q = np.quantile(boot_samples, [0.025, 0.975], axis=axis)
    lower = 2 * center - q[1]
    upper = 2 * center - q[0]
    return center - lower, upper - center


def format_asym(val, err_p, err_n, fmt=".3f", combine_if_equal=False):
    err_p_str = f"{err_p:{fmt}}"
    err_n_str = f"{err_n:{fmt}}"
    val_str = f"${val:{fmt}}"
    if err_p_str == err_n_str and combine_if_equal:
        return val_str + rf"\pm{{{err_p_str}}}$"
    return val_str + r"^{+" + err_p_str + r"}" + r"_{-" + err_n_str + r"}$"


def blinding_before_after(
    blinding_year,
    pair_years,
    quantity_idxs_pair,
    zs,
    boot=False,
    only_quantities_in_both=True,
    base_mask=None,
):
    # Split pairs into three groups based on the years of their two measurements:
    #   mask0: both measurements before blinding_year ("before" group)
    #   mask1: both measurements at or after blinding_year ("after" group)
    #   mask_across: one measurement on each side of blinding_year
    if base_mask is None:
        base_mask = np.ones(len(zs), dtype=bool)
    mask0 = base_mask & np.all(pair_years < blinding_year, axis=1)
    mask1 = base_mask & np.all(pair_years >= blinding_year, axis=1)
    assert np.all(
        pair_years[:, 0] <= pair_years[:, 1]
    )  # pairs are stored with earlier year first
    mask_across = (
        base_mask
        & (pair_years[:, 0] < blinding_year)
        & (pair_years[:, 1] >= blinding_year)
    )

    if only_quantities_in_both:
        # Restrict to quantities that have pairs in both the before and after groups,
        # so the comparison is not confounded by different quantities in each epoch.
        mask2 = shared_quantity_mask(mask0, mask1, quantity_idxs_pair)
        mask0, mask1, mask_across = mask0 & mask2, mask1 & mask2, mask_across & mask2

    boot_count = None
    if boot:
        # Bootstrap at the quantity level (not pair level) to preserve the
        # correlation structure within each quantity's pairs.
        quantities = np.unique(quantity_idxs_pair)
        sampled = np.random.choice(quantities, size=len(quantities), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        boot_count = defaultdict(int, zip(unique, counts))

    return np.array(
        [
            weighted_stats(zs[m], quantity_weights(m, quantity_idxs_pair, boot_count))
            for m in [mask0, mask1, mask_across]
        ]
    ).T  # shape (4, 3): 4 statistics x 3 groups (before, after, across)


def plot_datatypes(w, pair_types):
    # bar plot summing w for each pair type
    pair_types_unique = np.unique(pair_types)
    w_sum = np.zeros(len(pair_types_unique))
    for i, pair_type in enumerate(pair_types_unique):
        w_sum[i] = np.sum(w[pair_types == pair_type])
    plt.bar(pair_types_unique, w_sum)
    print(w_sum)
    plt.xlabel("Pair type")
    plt.ylabel("Weight")
    plt.title("Weight by pair type")
    plt.show()


# --- load data ---
_, pdg2025_both_dfs, _, _ = load_pdg_data()
# zs: array of |z_ij| for each pair; pair_years: (n_pairs, 2) array of measurement years;
# quantity_idxs_pair: (n_pairs,) array mapping each pair to its physical quantity index.

if USE_HIDDEN:
    hidden_df = pd.read_csv("../data/pdgH/pdg-hidden.csv")
    hidden_df["value"], hidden_df["error_positive"], hidden_df["error_negative"] = zip(
        *hidden_df["measurement"].apply(parse_measurement)
    )
    hidden_df = hidden_df[~hidden_df.value.isna()]
    hidden_df.rename(columns={"source_year": "year"}, inplace=True)
    hidden_dfs = list(hidden_df.groupby("node"))
    hidden_pdgids = [t[0] for t in hidden_dfs if len(t[1]) > 1]
    hidden_dfs = [t[1] for t in hidden_dfs if len(t[1]) > 1]
    for i, df in enumerate(pdg2025_both_dfs):
        pdgid = df.iloc[0]["pdgid"]
        if pdgid in hidden_pdgids:
            idx = hidden_pdgids.index(pdgid)
            pdg2025_both_dfs[i] = pd.concat([df, hidden_dfs[idx]], ignore_index=True)

pdgids = [df["pdgid"].iloc[0] for df in pdg2025_both_dfs]
type_map = pdgid_type_map()
# print(type_map)
pdgid_types = [type_map[pdgid] for pdgid in pdgids]

pair_dist = measurement_dist(
    pdg2025_both_dfs,
    ci=False,
    asym_error=["error_negative", "error_positive"],
    return_samples=True,
)

zs = pair_dist["zs"]
pair_years = pair_dist["pair_years"]
quantity_idxs_pair = pair_dist["quantity_idxs_pair"]
pair_types = np.array([pdgid_types[idx] for idx in quantity_idxs_pair])
zspace = pair_dist["zspace"]
normdist = pair_dist["norm"]  # survival function of |N(0,1)| evaluated on zspace
min_year, max_year = int(np.min(pair_years)), int(np.max(pair_years))
years = np.arange(min_year, max_year + 1)

base_mask = np.ones(len(zs), dtype=bool)
base_mask = base_mask & np.all(pair_years >= MIN_YEAR, axis=1)
if EXCLUDE_BR:
    base_mask = base_mask & (pair_types != "branching ratio")
if ONLY_BR:
    base_mask = base_mask & (pair_types == "branching ratio")


# --- 2D year-bin heatmap ---
# Each cell (i, j) shows the weighted average |z_ij| (or z_ij^2) for all pairs
# where the earlier measurement is from year years[i] and the later from years[j].
# Diverging colormap centered at the N(0,1) null expectation highlights
# years where measurements are over- or under-dispersed relative to expectations.
avg_zs_2d = np.full((len(years), len(years)), np.nan)
avg_z2s_2d = np.full((len(years), len(years)), np.nan)
for i, j in product(range(len(years)), range(len(years))):
    mask = (pair_years[:, 0] == years[i]) & (pair_years[:, 1] == years[j])
    if not mask.any():
        continue
    w = quantity_weights(mask, quantity_idxs_pair)
    stats = weighted_stats(zs[mask], w)
    avg_zs_2d[i, j] = stats[0]
    avg_z2s_2d[i, j] = stats[1]

X, Y = np.meshgrid(years + 0.5, years + 0.5)
div_cmap = plt.get_cmap("coolwarm")

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for ax, data, center, vmax, label in [
    (axs[0], avg_zs_2d, np.sqrt(2 / np.pi), 2, r"Average $|z_{ij}|$"),
    (axs[1], avg_z2s_2d, 1, 4, r"Average $z_{ij}^2$"),
]:
    pc = ax.pcolormesh(
        X,
        Y,
        data,
        cmap=div_cmap,
        # TwoSlopeNorm centers the colormap at the null expectation so deviations
        # in either direction are symmetric in color saturation.
        norm=TwoSlopeNorm(vmin=0, vcenter=center, vmax=vmax),
        shading="nearest",
    )
    cbar = fig.colorbar(pc, ax=ax, label=label, extend="max")
    cbar.ax.set_yscale("linear")
    cbar.ax.axhline(center, color="red")  # red line at null expectation
    ax.axhline(BLINDING_YEAR, color="black", linestyle="dashed", linewidth=1)
    ax.axvline(BLINDING_YEAR, color="black", linestyle="dashed", linewidth=1)
    ax.set_xlabel("Year of later measurement")
    ax.set_ylabel("Year of earlier measurement")
    ticks = ax.get_xticks()[1:-1]
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)

plt.tight_layout()
plt.savefig(f"../figs/blinding/{PREFIX}yearbin2d.pdf", bbox_inches="tight")
plt.close()


# --- 1D year-bin line plots ---
# For each year y, collect pairs where the later measurement is from year y
# (regardless of what year the earlier measurement is from), and compute
# weighted statistics. This shows how z-score statistics trend over time.
stats_by_year = []
for y in years:
    mask = base_mask & (pair_years[:, 1] == y)
    if not mask.any():
        stats_by_year.append([np.nan] * 4)
        continue
    stats_by_year.append(
        weighted_stats(zs[mask], quantity_weights(mask, quantity_idxs_pair))
    )
stats_by_year = np.array(stats_by_year).T  # shape (4, n_years)

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
for ax, vals, hline, title, ylabel in [
    (
        axes[0],
        stats_by_year[0],
        np.sqrt(2 / np.pi),
        r"Average $|z_{ij}|$ vs Year",
        r"Average $|z_{ij}|$",
    ),
    (
        axes[1],
        stats_by_year[1],
        1,
        r"Average $z_{ij}^2$ vs Year",
        r"Average $z_{ij}^2$",
    ),
    (
        axes[2],
        stats_by_year[2],
        1 - 0.6827,
        r"Fraction $|z_{ij}| > 1$ vs Year",
        r"Fraction $|z_{ij}| > 1$",
    ),
]:
    ax.plot(years, vals, color="black", linewidth=1)
    ax.axhline(hline, color="red", linewidth=1)  # red line at null expectation
    ax.set_xlabel("Year of later measurement")
    ax.set_title(title)
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig(f"../figs/blinding/{PREFIX}yearbin1d.pdf", bbox_inches="tight")
plt.close()


# --- Bootstrap before/after analysis ---
# Compute observed statistics for the before, after, and across groups.
results_center = blinding_before_after(
    BLINDING_YEAR, pair_years, quantity_idxs_pair, zs, base_mask=base_mask
)

# Run B bootstrap replicates, resampling quantities with replacement each time.
# The resulting array has shape (4, 3, B): 4 statistics x 3 groups x B replicates.
B = 4000
results_b = np.array(
    [
        blinding_before_after(
            BLINDING_YEAR,
            pair_years,
            quantity_idxs_pair,
            zs,
            boot=True,
            base_mask=base_mask,
        )
        for _ in tqdm(range(B))
    ]
).transpose(1, 2, 0)  # shape (4, 3, B)

# 95% CIs on each statistic for each group.
results_error_n, results_error_p = bootstrap_ci_errors(
    results_center, results_b, axis=2
)

# Differences between groups: "after minus before" and "across minus before/after".
# Bootstrap CIs on the differences are computed from the same bootstrap replicates,
# preserving the correlation between before and after estimates.
diffs_center = results_center[:, 1] - results_center[:, 0]  # after - before
diffs_error_n, diffs_error_p = bootstrap_ci_errors(
    diffs_center, results_b[:, 1, :] - results_b[:, 0, :]
)

diffs_center1 = results_center[:, 2] - results_center[:, 0]  # across - before
diffs_error_n1, diffs_error_p1 = bootstrap_ci_errors(
    diffs_center1, results_b[:, 2, :] - results_b[:, 0, :]
)

diffs_center2 = results_center[:, 2] - results_center[:, 1]  # across - after
diffs_error_n2, diffs_error_p2 = bootstrap_ci_errors(
    diffs_center2, results_b[:, 2, :] - results_b[:, 1, :]
)


# --- LaTeX tables ---
def make_latex_table(header, colspec, row_data):
    return (
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\hline\n"
        + header
        + "\\\\"
        + "\\hline\n"
        + "\n".join(row_data)
        + "\n\\hline\n"
        "\\end{tabular}"
    )


# Table 1: before vs after comparison, with difference column.
rows = [
    f"{ROW_LABELS[i]} & {EXPECTATIONS[i]}"
    f" & {format_asym(results_center[i,0], results_error_p[i,0], results_error_n[i,0])}"
    f" & {format_asym(results_center[i,1], results_error_p[i,1], results_error_n[i,1])}"
    f" & {format_asym(diffs_center[i], diffs_error_p[i], diffs_error_n[i])} \\\\"
    for i in range(len(ROW_LABELS))
]
print(
    make_latex_table(
        r"Statistic & Expectation & $<2000$ & $\geq 2000$ & Difference ",
        "l|rrr|r",
        rows,
    )
)

# Table 2: across-boundary pairs, with differences relative to before and after groups.
rows = [
    f"{ROW_LABELS[i]} & {EXPECTATIONS[i]}"
    f" & {format_asym(results_center[i,2], results_error_p[i,2], results_error_n[i,2])}"
    f" & {format_asym(diffs_center1[i], diffs_error_p1[i], diffs_error_n1[i])}"
    f" & {format_asym(diffs_center2[i], diffs_error_p2[i], diffs_error_n2[i])} \\\\"
    for i in range(len(ROW_LABELS))
]
print(
    make_latex_table(
        r"Statistic & Expectation & Across & Difference with $\leq 2000$ & Difference with $>2000$ ",
        "l|rr|rr",
        rows,
    )
)


# --- Before/after distribution plot ---
# Recompute masks (same logic as blinding_before_after, but kept explicit here
# for clarity when constructing the survival curves for the plot).
mask0 = base_mask & np.all(pair_years < BLINDING_YEAR, axis=1)
mask1 = base_mask & np.all(pair_years >= BLINDING_YEAR, axis=1)
mask_across = (
    base_mask & (pair_years[:, 0] < BLINDING_YEAR) & (pair_years[:, 1] >= BLINDING_YEAR)
)
mask2 = shared_quantity_mask(mask0, mask1, quantity_idxs_pair)
mask0, mask1, mask_across = mask0 & mask2, mask1 & mask2, mask_across & mask2

print(np.sum(mask0), np.sum(mask1), np.sum(mask_across))
print(len(np.unique(quantity_idxs_pair[mask2])))

fig, axs = plt.subplots(1, 2, figsize=(6, 2.5))
colors = ["grey", "black", "blue"]
labels = [r"$\leq 2000$", r"$> 2000$", "across"]
for mask, color, label in zip([mask0, mask1, mask_across], colors, labels):
    w = quantity_weights(mask, quantity_idxs_pair)
    # Weighted empirical survival function: P(|Z| > z) at each point in zspace.
    surv = survival_curve(zs[mask], w, zspace)
    for ax in axs:
        ax.plot(zspace, surv, color=color, linewidth=1.5, label=label)

for ax in axs:
    ax.plot(
        zspace,
        normdist,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="$|N(0,1)|$",
    )
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$P(|Z|>z)$")

# Left panel: zoom in on z < 2 to show the bulk of the distribution.
# Right panel: out to z = 6 to show tail behavio/r.
axs[0].set_xlim(0, 2)
axs[0].set_ylim(2e-2, 1)
axs[1].set_xlim(0, 6)
axs[1].set_ylim(1e-4, 1)
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].legend(frameon=False)

# Read table number from .aux file so we can annotate the plot with a cross-reference.
with open("../technote/main.aux") as f:
    aux = f.read()
match = re.search(r"\\newlabel{tab:blinding-quadrants}{{([^}]*)}", aux)
table_number = match.group(1) if match else "?"

# Annotate the z=1 and z=2 thresholds to point readers to the quantitative table.
axs[0].annotate(
    "",
    xy=(1, 1 - 0.6827 + 0.05),
    xytext=(1.2, 0.5),
    arrowprops=dict(arrowstyle="->", color="black"),
)
axs[0].annotate(
    "",
    xy=(1.99, 0.095),
    xytext=(1.6, 0.5),
    arrowprops=dict(arrowstyle="->", color="black"),
)
axs[0].text(
    1.4, 0.5, rf"See Table {table_number}", ha="center", va="bottom", fontsize=10
)
axs[0].add_patch(
    plt.Rectangle(
        (0.975, 0.2), 0.05, 0.2, edgecolor="none", facecolor="blue", alpha=0.2
    )
)
axs[0].add_patch(
    plt.Rectangle(
        (1.975, 0.01), 0.05, 0.09, edgecolor="none", facecolor="blue", alpha=0.2
    )
)

plt.tight_layout()
rect, lines = axs[1].indicate_inset_zoom(axs[0])
for line in lines:
    line.set_visible(True)
    line.set_zorder(10)
axs[0].set_zorder(100)

plt.savefig(f"../figs/blinding/{PREFIX}beforeafterdist.pdf", bbox_inches="tight")
plt.savefig(
    f"../figs/blinding/{PREFIX}beforeafterdist.png", bbox_inches="tight", dpi=300
)
plt.close()


# --- Blinded vs non-blinded collaborations ---
# For each pair (i, j), check whether each measurement's collaboration is in
# the blinded or non-blinded list. pair_techniques has shape (n_pairs, 2).
blind = np.isin(pair_dist["pair_techniques"], BLINDED_COLLABS)
notblind = np.isin(pair_dist["pair_techniques"], NOT_BLINDED_COLLABS)

# mask_blind: both measurements from blinded collaborations
# mask_notblind: both from non-blinded collaborations
# mask_across: one from each (mixed pair)
mask_blind = base_mask & np.all(blind, axis=1)
mask_notblind = base_mask & np.all(notblind, axis=1)
mask_across = base_mask & (
    (blind[:, 0] & notblind[:, 1]) | (blind[:, 1] & notblind[:, 0])
)
# Restrict to quantities with pairs in both blinded and non-blinded groups.
mask2 = shared_quantity_mask(mask_blind, mask_notblind, quantity_idxs_pair)
mask_blind, mask_notblind, mask_across = (
    mask_blind & mask2,
    mask_notblind & mask2,
    mask_across & mask2,
)

print([np.sum(m) for m in [mask_blind, mask_notblind, mask_across]])

plt.figure(figsize=(4, 2))
colors = ["black", "grey", "blue"]
labels = [r"Blinded collaborations", r"Non-blinded collaborations", "across"]
for mask, color, label in zip([mask_blind, mask_notblind, mask_across], colors, labels):
    w = quantity_weights(mask, quantity_idxs_pair)
    surv = survival_curve(zs[mask], w, zspace)
    plt.plot(zspace, surv, color=color, linewidth=1.5, label=label)

plt.plot(
    zspace, normdist, color="red", linestyle="dashed", linewidth=2, label="$|N(0,1)|$"
)
plt.legend(frameon=False)
plt.xlim(0, 2.5)
plt.ylim(0, 1)
plt.ylabel(r"$P(|Z| > z)$")
plt.xlabel("$z$")
plt.title("Blinded vs non-blinded collaborations")
plt.tight_layout()
plt.savefig(
    f"../figs/blinding/{PREFIX}blinded_nonblinded_dist.pdf", bbox_inches="tight"
)
plt.savefig(
    f"../figs/blinding/{PREFIX}blinded_nonblinded_dist.png",
    bbox_inches="tight",
    dpi=300,
)
plt.close()

plot_datatypes(w, pair_types[mask])

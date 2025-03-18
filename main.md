---
title: Aggregating experiments with weaker assumptions
exports:
  - format: pdf
    template: arxiv_two_column
    output: exports/test.pdf
authors:
  - name: Anthony Ozerov
    affiliations:
      - University of California, Berkeley
    orcid: 0000-0001-8500-3566
    email: ozerov@berkeley.edu
license: CC0-1.0
abstract: |
    Many studies involve aggregating multiple estimates of a quantity. We introduce a new method, the "Binomial Method" (BM) for doing this, which only assumes that the estimates each independently overestimate or underestimate the target quantity with one-half probability. In simulations, confidence intervals generated under BM better achieve target coverage than existing methods like Random Effects and the Birge Ratio method. BM results for the gravitational constant and the Planck constant widen currently accepted uncertainties.
---


# Introduction

Combining multiple estimates of a quantity into one is common across the sciences. In medicine and the social sciences, meta-analyses find studies/experiments which estimate the same quantity (e.g. the treatment effect of a medicine) and combine them into one estimate, with associated uncertainties. In Earth science, "intercomparison" studies take models from different teams and study their differences, sometimes attempting to reconcile them and produce one better estimate. In physics, the values of fundamental constants obtained in different experiments must also be somehow reconciled.

One key issue that arises is systematic uncertainty, where an estimate of a quantity may have some error not due to noise which can't be reduced by collecting a larger sample / running the experiment for longer. This error can be due to issues in the experiment like a non-randomly sampled population, or an instrument which is miscalibrated, or analysis issues like approximations and model imperfections. Empirically, this can be seen when results from different experiments/teams/methods are far outside of each other's error bars. Aggregating multiple estimates must therefore not only account for the noise, but also somehow model the systematic uncertainties in the different estimates. This leads to the counterintuitive fact that, sometimes, combining multiple estimates *increases* our uncertainty, rather than decreasing it---this is because the level of systematic error is not apparent in only one estimate, and becomes apparent when we see more.

## Setup

Suppose we are given $n$ estimates $y_1,\ldots,y_n$ of a number $\theta$, from $n$ experiments/teams/methods. Each of these estimates has error from two sources: **systematic** and **statistical** (i.e. noise). (For instance, we can have a data-generating process where $y_i$ is actually a noisy estimate of $\theta_i$, which is what experiment $i$ would see if it had no noise; because of systematic error, in general $\theta_i\neq \theta$).

Suppose we are further given a standard deviation $\sigma_i$ for each $y_i$. Experimenter $i$ believes that $\sigma_i$ is a good estimate for the uncertainty in $y_i$ due to noise.

In this setting, how can we learn something about $\theta$? Without making any assumptions, we can't learn anything (for instance, we could have $\theta_i=\theta-a\ \forall i$, in which case we are unable to recover $\theta$ without knowing something about $a$).

# Existing approaches

Let's take a look at two existing statistical approaches used to deal with systematic error.

## Random Effects

The Random Effects model and method originated in the medical meta-analysis literature [@dersimonian1986meta]. It is now common meta-analyses in medicine and the social sciences, but appears to be uncommon in the physical sciences.

### Assumptions

The Random Effects model places some distributional assumptions upon our setting:

1.  $y_i|\theta_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta_i,\sigma_i^2)$

2.  $\theta_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(\theta,\tau^2)$

$\tau^2$ is a new parameter introduced to model the spread of the systematically-biased experiment values $\theta_1,\ldots,\theta_n$ around the true value $\theta$.

From these assumptions, we get $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta,\sigma_i^2+\tau^2)$. Effectively, the systematic uncertainty and noise are additive.

### Inference

Inference on $\theta$ can be done in a number of ways under this model. The original DerSimonyan and Laird procedure (DL) first estimates $\tau^2$ using the method of moments, then uses this as a plug-in value to estimate $\theta$ and a corresponding confidence interval Under a Normal approximation [@dersimonian1986meta]. Variants exist, such as using maximum likelihood for $\tau^2$ [@dersimonian1986meta; @jackson2010does].

A later procedure, Hartung-Knapp-Sidik-Jonkman (HKSJ), changed the uncertainty estimation in DL to better account for the fact that our estimate of $\tau^2$ is uncertain [@hartung1999alternative; @sidik2002simple]. HKSJ has been shown to generally achieve target coverage far more consistently than DL [@inthout2014hartung] and to generally give wider confidence intervals [@wiksten2016hartung]. In this paper we will use the HKSJ procedure as we also found it to generally perform better than DL in our simulations.

Other works instead perform Bayesian inference, by placing informative or uninformative priors on $\tau$ and $\theta$ [@sutton2001bayesian].

### Extensions

The Random Effects model can and has been extended to the case where there are correlations between experiments in $\theta_i$'s or $y_i|\theta_i$'s.

The model can also be extended to include information from study-level variables (e.g. year of publication, method used) in the distribution of $\theta_i$ to explain more of the spread in study-level effects.

## Birge Ratio

The Birge ratio originated as a way to measure the inconsistency of different estimates and their uncertainties (a quantitative version of "these confidence intervals don't overlap"). The Birge Ratio method (BR) is effectively what is now used by CODATA (under the name "Least-Squares Adjustment" [@tiesinga2021codata]) when they combine estimates of fundamental constants from multiple experiments. (Historically this has not always been the case, as in the 2002 adjustment's determination of $G$ [@mohr2005codata])

### Assumptions

1.  $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta,c^2\sigma_i^2)$

The idea is that the uncertainties reported in the different experiments are too small---as they only represent noise, and not systematic uncertainty. So we can just expand the uncertainties by a scaling factor.

### Inference

To do inference, we estimate $c$ and then directly estimate $\theta$ by plugging in $\hat c$. Note that in practice, when estimating $c^2$, it is often clipped to be at least $1$.

As with the Random Effects model, we can also do inference by placing informative or uninformative priors on $c^2$ and $\theta$ [@bodnar2014adjustment].

### Extensions

Just like the random effects model, the Birge ratio method can be extended to the case where there are correlations between experiments. The CODATA analyses often have correlated experiments, so they use this extension.

# Proposed Method

We can call this method the "Binomial Method."

## Assumptions

Let's make the following assumptions:

:::{important} Assumptions of the Binomial Method
1.  $P(y_i>\theta)=0.5$

2.  The event $y_i>\theta$ is independent for each $i$.
:::

In words, Assumption 1 says that each experiment has a one-half chance of being an overestimate of the truth (and, correspondingly, a 1/2 chance of being an underestimate). We don't get much from this assumption alone, because we could have all the experiments be correlated. So we need the additional Assumption 2 that the experiments are independently overestimates or underestimates.

The $y_i$'s being independent for each $i$ is sufficient but not necessary to satisfy Assumption 2. A setting where Assumption 2 holds and the $y_i$'s are not independent would surely be very contrived, so for practical purposes we can consider Assumption 2 not satisfied when the $y_i$'s are not independent.

Let $K$ be the number of experiments which yield an underestimate. From Assumptions 1 and 2, we get:
\begin{equation*}K\sim\mathrm{Binomial}(n,0.5)\end{equation*}

## Inference

For simplicity, let's suppose $y_i\neq y_j$ for all $i\neq j$, and that $y_i\neq \theta$ for all $i$.

Let's consider the order statistics $y_{(1)}<\ldots< y_{(n)}$ (simply the sorted $y_1,\ldots,y_n$). For simplicity of notation, let $y_{(n+1)}=\infty$.

If $K=k$, we know that $y_{(k)}$ is an underestimate and therefore $y_{(k)}< \theta$. We also know that $y_{(k+1)}$ is an overestimate, and therefore $\theta < y_{(k+1)}$. In fact, we can say:

-   Whenever $K\leq k$, we know $y_{(k+1)}$ is an overestimate, and therefore $\theta<y_{(k+1)}$.

-   Whenever $\theta<y_{(k+1)}$, $y_{(k+1)}$ is an overestimate, and therefore $K\leq k$.

In short, we have
\begin{equation*}K\leq k\Longleftrightarrow \theta< y_{(k+1)},\end{equation*}
from which we directly get
\begin{equation*}P(\theta<y_{(k+1)})= P(K\leq k).\end{equation*}
The uncertainty in the LHS is over $y_{(k+1)}$. $P(K\leq k)$ can be directly calculated from the Binomial CDF.

How can we turn these nice facts into a $1-\alpha$ confidence interval? Say we want a symmetric interval. For our lower bound $l$, pick the smallest $j$ such that $P(K\leq j)\leq \alpha/2$, and let the lower bound be $l=y_{(j+1)}$. We will get that:
\begin{equation*}P(\theta<l)=P(\theta<y_{(j+1)})=P(K\leq j)\leq \alpha/2 .\end{equation*}
For our upper bound $u$, pick the largest $j$ such that $P(K\geq j)\leq \alpha/2$, and let the upper bound be $u=y_{(j)}$. We
will get that:
\begin{equation*}P(\theta>u)=P(\theta>y_{(j)})=P(K\geq j)\leq \alpha/2\end{equation*}
The interval $[l,u]$ will have level $1-\alpha$.

Note that this method doesn't require knowledge of the $\sigma_i^2$'s, so them being mis-specified won't affect results.

See [](#binomial-method) for a minimal Python implementation of this inference procedure.

```{code} python
:label: binomial-method
:caption: Minimal Python code for the Binomial Method's inference. The main thing to be careful with in implementation is correct indexing.

import numpy as np
from scipy.stats import binom
def binomial_method(values, p=0.5, target=0.15865):
    n = len(values)
    cdf = binom.cdf(np.arange(n+1), n, p)

    lower_idx = np.argmax(cdf > target)-1
    upper_idx = np.argmax(cdf >= 1-target)

    return values[lower_idx], values[upper_idx]
```

## Extensions

### Relaxing Assumption 1

Saying that the probability of an overestimate is exactly 0.5 could be a strong assumption---though it is still much weaker than the assumptions of the Random Effects or Birge ratio methods. Here are some ways to relax this:

-   Assume instead "the probability of an overestimate is between $p_1$ and $p_2$," where $p_1<p_2$. Then, we can compute $l$ under $p_2$ and $u$ under $p_1$ (a larger $p$ shifts the interval downwards, because it implies more estimates are too high).

-   Assume instead "the probability $p$ of an overestimate follows a $\mathrm{Beta}(\alpha,\beta)$ distribution." Then $K\sim\mathrm{BetaBinomial}(n,\alpha,\beta)$, and we can do everything else the same.

Either method will yield a wider interval which takes into account our less-confident assumptions about $p$.

### Relaxing Assumption 2

Saying that the events $y_i>\theta$ are independent could also be a strong assumption. For example, there may be a grouping effect among studies. To relax this, we can model $K$ as another known or derived probability distribution (e.g., again, a Beta-Binomial) or one estimated from simulations under a custom data-generating process. However, it does not seem easy to reason about what the relevant conditional probabilities might be, even for two estimates (this would be the probability $P(y_i>\theta|y_j>\theta),\ i\neq j$), without placing additional distributional assumptions on our setting.

## Analysis

The assumptions of the Binomial Method are far weaker than the assumptions of the previous methods. In fact, the assumptions of RE and BR imply Assumptions 1 and 2. So a method using only Assumptions 1 and 2 should work well under data from either model, but not vice-versa.

Our assumptions and inference also do not at all involve the $\sigma_i$'s, making our method completely insensitive to their mis-specification.

Unlike most inferential methods for RE and BR, our inference does not involve a prior or any distributional approximations. Under Assumptions 1 and 2, inference will be exact for any value of $n$, and coverage which is at least at the target level is guaranteed.

One caveat is that the Binomial distribution is discrete, so for finite $n$ the tail probabilities only take on a set of discrete values. So, without playing tricks like randomly shrinking or collapsing the interval, we cannot achieve exactly 0.6827 ($1\sigma$) coverage, but will always achieve coverage $\geq 0.6827$. See Table [](#table-n) for examples of values of $n$ and the smallest coverage level $\geq0.6827$ which can be achieved exactly---usually, it is not too much higher.

# Simulation study

Let's do some simulations to benchmark the Binomial method (BM) against RE and BR. In every simulation, we will have these parameters:


-   $n$: how many estimates we get.

-   $\theta=0$: the truth we want to contain in our interval

-   $\alpha/2$: our target tail probability.

-   $\sigma_i^2\sim\mathrm{Exp}(1)$: the standard deviations of the
    noise (observed)

-   $\tau$: a parameter we control to vary the level of systematic error
    or underestimation of $\sigma_i$'s. A larger $\tau$ indicates larger
    systematic errors or underestimation of $\sigma_i$'s.

All of the methods in this study are location- and scale-invariant, so we don't lose anything of interest by fixing $\theta=0$ and the scale of the $\sigma_i^2$'s. The different experiments will vary $n$, the distribution of the systematic errors, and their scale relative to the $\sigma_i^2$'s.

Because the Binomial distribution is discrete, for arbitrary choice of $(n,\alpha)$ the Binomial method we described will be conservative. Here are two ways to achieve exact or close coverage for arbitrary $\alpha$:

-   Trivially, we could randomly shrink or collapse the interval to make it exact, but that would be silly.

-   We could make the interval not have symmetric tail probabilities. This would let us sometimes get closer to a $1-\alpha$ coverage, but the choice of which tail should be heaver in any given scenario is arbitrary.

So let's stick with the Binomial method as described. To provide a reasonable assessment of the its exactness and interval lengths, we will pick $\alpha$ so that the BM we propose can provide exact $1-\alpha$ coverage. Because $1\sigma$ intervals are commonly used in the sciences, we will pick the smallest $\alpha$ such that $1-\alpha\geq 0.6827$ and the BM can be exact.

For $n=10$, for example, this means targeting a coverage of $0.891$ (narrower symmetric intervals could achieve $0.656$---just below our stated minimum of $0.6827$).

For a choice of $\alpha$, we use the corresponding $z_{\alpha/2}$ to expand the intervals of the Random Effects and Birge methods. See Table [](#table-n) for our choices of $n$ and the corresponding target level, tail probability, and $z$-score.

:::{table} Chosen level and corresponding $\alpha/2$ tail probability and $z$-score for different $n$. $n$ is evenly spaced on a logarithmic scale.
:label: table-n
:align:center
| $n$ | Level | $\alpha/2$ | $z_{\alpha/2}$|
|------|-------|------------|----------------|
| 3    | 0.750 | 0.125      | 1.150          |
| 10   | 0.891 | 0.055      | 1.601          |
| 31   | 0.719 | 0.141      | 1.078          |
| 100  | 0.729 | 0.136      | 1.100          |
| 316  | 0.715 | 0.143      | 1.069          |
| 1000 | 0.703 | 0.148      | 1.044          |

:::

## Simulation settings

**Random Effects model**

-   $\theta_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,\tau^2)$: the systematic errors

-   $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta_i,\sigma_i^2)$: the estimates (observed)

**Birge**

-   $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(0,\tau^2\sigma_i^2)$: the estimates (observed)

**Random Effects with Outliers**

-   $\theta_i\overset{\mathrm{iid}}{\sim}\tau\cdot \mathrm{Cauchy}(0,1)$: the systematic errors

-   $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta_i,\sigma_i^2)$: the estimates (observed)

This is effectively Random Effects, just with a different distribution for $\theta_i$ that makes outliers more likely.

**Adversarial**

-   $y_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(\tau,\sigma_i^2)$: the estimates (observed)

In this setting, all estimates are systematically offset by $\tau$ from the truth. As discussed in the first section, in principle there is no way to obtain the truth without knowing something about $\tau$, so all of the methods should fail. But we will see which methods fail faster.

**Correlated Random Effects**

-   $\boldsymbol{\theta}\sim \mathcal{N}_n(\mathbf{0},\Sigma)$ where $\Sigma=\tau^2(0.8\mathbf{I}_n+0.2)$

-   $y_i\overset{\mathrm{ind}}{\sim}\mathcal{N}(\theta_i,\sigma_i^2)$

This is like Random Effects, but where the systematic errors are correlated between experiments. It will violate the assumptions of all the methods we have presented. Note that due to the computational difficulty of sampling a high-dimensional multivariate Normal, we will not experiment with $n>100$ under this model.

## Simulation results

:::{figure}
:label: sim-results
![](figs/performance_random_effects.pdf)
![](figs/performance_birge.pdf)
![](figs/performance_random_effects_outliers.pdf)

Simulation results under the three settings satisfying our assumptions. The darkest color is $n=1000$, and the lightest color is $n=3$.
:::

We show results for the settings satisfying our assumptions (Random Effects, Birge, Random Effects with Outliers) in [](#sim-results). Since the target coverage level is varied across simulations, we report the difference in achieved coverage and target coverage. Here are some observations.

-   The Binomial method perfectly achieves target coverage in every single scenario. This is not surprising given its mathematical setup, and that every single scenario satisfies BM's assumptions.

-   Under the Random Effects model, RE yields better coverage with smaller intervals when the systematic errors are small, but does not achieve coverage when systematic errors are comparable to or larger than noise. This is surprising until we remember that inference for the Random Effects model is approximate, not exact. BR performs very poorly, espeially for large $n$.

-   Under the Birge model, note that it is slightly unrealistic to have $\tau<1$, because that implies that the $\sigma_i$'s given to us are actually *overestimates* of the true standard deviation. Considering the case where $\tau\geq 1$, the BR method works well, as we would expect, though does it does not achieve target coverage when $n$ is small (as inference is approximate and not exact). BR also offers far narrower intervals than BM. RE works decently well; maintains at least target coverage, but offers larger intervals for large $n$ than Binomial and BR for roughly $\tau>3$. For context, in the gravitational constant data shown later [@tiesinga2021codata], we estimate $\tau\approx 3.6$ under the Birge model.

-   Under Random Effects with Outliers, which satisfies the assumptions of neither RE nor BR, BR works poorly, while RE is fairly competitive with BM (though it yields intervals larger by orders of magnitude for large $n$ with $\tau>1$). This difficulty in this setting is outliers, especially at large $n$, which pull or greatly expand the RE and BR intervals. As expected, BM still achieves perfect coverage in this setting. In general, BM is robust to outliers as it only depends on the ranking of the experiments, not their relative magnitudes.

:::{figure}
:label: sim-results2
![](figs/performance_adversarial.pdf)
![](figs/performance_random_effects_corr.pdf)

Simulation results under the two settings not satisfying our assumptions. The darkest color is $n=1000$, and the lightest color is $n=3$. Note that in the Correlated Random Effects setting, we don't do simulations for $n>100$ due to computational difficulties.
:::

[](#sim-results2) contains results for the settings not
satisfying our assumptions (Adversarial and Correlated Random Effects).

-   As $\tau$ increases, the Binomial method maintains target coverage for longer than RE and BR for all values of $n$.

-   For small $n$, all methods are less sensitive to the these particular assumption violations.

# Results on real data

We ought to run a comparison on some real data. Let's see what happens with {abbr}`CODATA (Committee on Data of the International Science Council)` estimates of physical constants. Because physical constants aren't known exactly, we don't have a ground truth, but we can at least see the behavior of the different methods with real experimental data.

## Gravitational constant

:::{figure} ./figs/G0.pdf
:label: fig:G

Results of different aggregation methods for estimates of the gravitational constant $G$ [@tiesinga2021codata, Table XXIX]. Error bars are $1\sigma$ intervals using the $\sigma_i$'s reported by each study
:::

We get data from 16 experiments estimating the gravitational constant $G$ from the 2018 CODATA adjustment [@tiesinga2021codata, Table XXIX]. Results are shown in [](#fig:G). Note that for $n=16$, the nominal coverage level of the Binomial interval is $0.7900$. The nominal coverages of the other intervals under their respective models are $0.6827$.

We see that the CODATA interval is slightly wider than that given by our application of BR. CODATA uses BR too, but there are two differences:

-   They take into account small correlations between some pairs of experiments.[^pairs]

-   Their inference on $\tau$, effectively the expansion factor for the $\sigma_i$'s is heuristic, and they choose $\tau=3.9$ to "decrease the residuals to below two" [@tiesinga2021codata].

[^pairs]:(NIST-82, LANL-97), (HUST-05, HUST-09), and (HUST-09, HUST$_\text{T}$-18) are correlated [@tiesinga2021codata].

The CODATA and BR intervals are pulled to the right by the small $\sigma_i$'s on many of the rightmost points. The RE interval is just about disjoint with the CODATA and BR intervals---this is because it models systematic errors additively, rather than multiplicatively. The Binomial interval is wider than any of the other intervals. Note how it is supported by the HUST-09 and UZur-06 points, and has 5 points outside it on its left, and 5 points outside on the right.

## Planck constant

:::{figure} ./figs/h0.pdf
:label: fig:h
Results of different aggregation methods for estimates of the Planck constant $h$ [@mohr2012codata, Table 26]. Error bars are $1\sigma$ intervals using the $\sigma_i$'s reported by each study.
:::

We get data from 11 experiments estimating the Planck constant $h$ from the 2010 CODATA adjustment [@mohr2012codata, Table 26].[^si]

[^si] Since the 2019 SI unit redefinition, the SI units are defined, in part, based on the Planck constant, so there is no longer "uncertainty" in it in terms of SI units. But the dataset still offers an interesting case study.

Results are shown in [](#fig:h). Note that for $n=11$, the nominal coverage level of the Binomial interval is $0.7734$. The nominal coverages of the other intervals under their respective models are $0.6827$.

We see again that the CODATA interval is wider than that given by our BR---this is because CODATA take into account some correlations between data points. In the Planck constant data, the BR interval is contained within RE, and RE is contained with the Binomial interval. We see that the Binomial interval is supported by points NMI-89 and IAC-11, with 3 points lying outside to either side.

# Discussion

The Binomial method is strong because it:

-   Makes weaker assumptions which are implied by many existing common assumptions.

-   Yields confidence intervals which exactly achieve their nominal coverage level for any value of $n$ and for any data-generating process which satisfies the relatively weak assumptions.

-   Is insensitive to problems in the $\sigma_i$ uncertainties provided for each underlying estimate.

-   Is robust to outliers, because it only depends on the ranking of the $y_i$'s, not their relative magnitudes or residuals.

It is weak in that it:

-   Cannot exactly achieve an arbitrary target coverage level for finite $n$ due to the discreteness of the Binomial distribution. (But it will be conservative, and anyway for any $n$ one can pick from a set of coverage levels which can be achieved exactly).

-   Yields wider intervals than needed if we know more about the data-generating process---methods tailored to that DGP can achieve coverage with narrower intervals.

-   Does not take advantage of the information in the $\sigma_i$'s.

# Targeting the same coverage level on real data

:::{figure}
:label: fig:Gh-wider
![](figs/G1.pdf)
![](figs/h1.pdf)

Since the Binomial method can target only a discrete set of coverages, for the $G$ data it will (under the assumptions) achieve coverage 79.0%, and 77.3% for the $h$ data. For comparison, in this figure the intervals from CODATA, RE, and BR are expanded to target these coverage levels. The CODATA, RE, and BR intervals from [](#fig:G),[](#fig:h) have been scaled them by the corresponding $z_{\alpha/2}$.
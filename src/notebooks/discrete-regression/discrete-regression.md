<a href="https://colab.research.google.com/github/rameshaditya-me/Easy-Classical-ML-DL/blob/main/src/notebooks/discrete-regression/discrete-regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Generalized Linear Models
---

**Generalized Linear Models (GLMs)** extend ordinary linear regression to many kinds of response variable. The same linear structure in the predictors can model continuous, discrete, binary, ordinal, categorical, and count outcomes — the choice of **link function** determines how $\beta^\top X$ connects to the distribution of $Y$.

The key idea is twofold. First, a link $g$ maps the linear predictor $\beta^\top X$ to a parameter of the outcome distribution (e.g. the mean). Second, the variance of each observation may depend on that predicted value — heteroscedasticity is built into the model rather than treated as a nuisance.

A GLM has three parts:

1. **Random component.** The outcome $Y$ is drawn from an **exponential-family** distribution (Gaussian, Bernoulli, Poisson, etc.).

2. **Systematic component.** One or more covariates $X_0, \ldots, X_d$ enter linearly through $\beta^\top X$.

3. **Link function.** A function $g$ connects the systematic component to the mean (or natural parameter) of the random component: $g(\mathbb{E}[Y \mid X]) = \beta^\top X$.

### Exponential Family of Probability Distributions
---
Any probability distribution that can be written in the following parametric form belongs to the **exponential family**:

$$
p_{\theta}(y) = h(y)\,\exp\!\left(\eta(\theta)\,T(y) - A(\theta)\right)
$$

where $h(y)$ is the **base measure**, $\eta(\theta)$ is the **natural parameter**, $T(y)$ is the **sufficient statistic**, and $A(\theta)$ is the **log-partition function**.

#### Gaussian with unknown mean and known variance

$$
p(y) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

Expanding $(y-\mu)^2$ and matching the exponential-family form:

$$
p(y) =
\underset{\text{base measure}}{\overbrace{
  \frac{1}{\sqrt{2\pi\sigma^2}}\,
  \exp\!\left(-\dfrac{y^2}{2\sigma^2}\right)
}^{h(y)}}\;
\exp\!\left(
  \underset{\text{natural parameter}}{\overbrace{\dfrac{\mu}{\sigma^2}}^{\eta(\theta)}}\,
  \underset{\text{sufficient statistic}}{\overbrace{y}^{T(y)}}
  -
  \underset{\text{log-partition}}{\overbrace{\dfrac{\mu^2}{2\sigma^2}}^{A(\theta)}}
\right)
$$

Here $\theta = \mu$, so $\eta(\mu) = \mu/\sigma^2$ and $T(y) = y$.

#### Binomial distribution with a known number of trials $n$

Let $Y \in \{0, 1, \ldots, n\}$ count the number of successes in $n$ independent trials, each with success probability $p$:

$$
P(y) = \binom{n}{y}\, p^{y}(1-p)^{n-y}
$$

Rewriting in log form and collecting terms in $y$:

$$
P(y) = \binom{n}{y}\exp\!\left(y\log\frac{p}{1-p} + n\log(1-p)\right)
$$

Matching the exponential-family form $p_{\theta}(y) = h(y)\exp(\eta(\theta)T(y) - A(\theta))$:

$$
P(y) =
\underset{\text{base measure}}{\overbrace{
  \binom{n}{y}
}^{h(y)}}\;
\exp\!\left(
  \underset{\text{natural parameter}}{\overbrace{\log\!\left(\dfrac{p}{1-p}\right)}^{\eta(\theta)}}\,
  \underset{\text{sufficient statistic}}{\overbrace{y}^{T(y)}}
  -
  \underset{\text{log-partition}}{\overbrace{\left(-n\log(1-p)\right)}^{A(\theta)}}
\right)
$$

Here $\theta = p$, so $\eta(p) = \log\!\big(p/(1-p)\big)$ (the **log-odds**) and $T(y) = y$. Equivalently, writing $A$ in terms of the natural parameter alone:

$$
A(\eta) = n\log\!\left(1 + e^{\eta}\right)
$$

The **Bernoulli** distribution is the special case $n = 1$; logistic regression uses this exponential-family form with the logit link $\eta = \beta^\top x$.



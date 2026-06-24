import math
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

JITTER = 1e-5  # diagonal nugget; also keeps K PD when rating-vectors repeat


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def ard_rbf(X1, X2, lengthscales, variance):
    """ARD squared-exponential (RBF) kernel.

    X1: (N, D), X2: (M, D), lengthscales: (D,), variance: scalar -> (N, M).

    Squared distances are formed via the ||a-b||^2 = a^2 + b^2 - 2ab identity
    rather than torch.cdist. cdist takes a sqrt internally, whose gradient is
    undefined at distance 0 -- and your discrete 1..5 features GUARANTEE
    coincident points (repeated rating vectors). The identity below keeps the
    gradient smooth everywhere, so NUTS does not hit NaNs on those repeats.
    """
    X1s = X1 / lengthscales
    X2s = X2 / lengthscales
    x1sq = (X1s ** 2).sum(-1, keepdim=True)        # (N, 1)
    x2sq = (X2s ** 2).sum(-1, keepdim=True).t()    # (1, M)
    sqdist = (x1sq + x2sq - 2.0 * (X1s @ X2s.t())).clamp(min=0.0)
    return variance * torch.exp(-0.5 * sqdist)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def gpc_model(X, y=None, base_rate=0.1, linear_mean=False):
    """Whitened-latent GP classifier. base_rate sets the mean prior location.

    Every sample site inherits X's dtype (via X.new_*). Pass X as float64 and
    the whole model runs in float64 -- which the Cholesky needs to stay stable
    -- without globally mutating torch's default dtype.

    linear_mean: if False (default) the latent mean is a constant logit(base_rate),
    so predictions revert to the base rate far from any analog. If True the mean is
    an AFFINE function m(x)=b0 + x@b1, so OFF-SUPPORT the GP reverts to this *trend*
    (P -> 0/1 along the slope) instead of flattening to the base rate. Opt-in: the
    default keeps every existing result identical.
    """
    N, D = X.shape
    mean_loc = math.log(base_rate / (1.0 - base_rate))

    # --- hyperpriors (weakly informative; sanity-check with a prior predictive)
    # features live on a 1..5 scale, so a lengthscale around 2 is sensible
    lengthscales = pyro.sample(
        "lengthscales",
        dist.LogNormal(X.new_full((D,), math.log(2.0)), X.new_tensor(0.5)).to_event(1),
    )
    # signal variance on the LOGIT scale; median 1 => latent swings ~+/-1 logit
    kernel_var = pyro.sample("kernel_var",
                             dist.LogNormal(X.new_tensor(0.0), X.new_tensor(0.75)))
    # mean function; identifiable separately from the GP because RBF correlations
    # decay, so the latent cannot supply an arbitrary trend far out
    if linear_mean:
        # b0 anchors x=0 (woe=0 = "no evidence") at the base rate; b1 lets the mean
        # trend with the features so extrapolation continues toward 0/1, not flat.
        b0 = pyro.sample("mean_intercept",
                         dist.Normal(X.new_tensor(mean_loc), X.new_tensor(1.5)))
        b1 = pyro.sample("mean_slope",
                         dist.Normal(X.new_zeros(D), X.new_full((D,), 2.0)).to_event(1))
        mean = b0 + X @ b1                              # (N,)
    else:
        mean = pyro.sample("mean", dist.Normal(X.new_tensor(mean_loc), X.new_tensor(1.5)))

    # --- whitened latent
    K = ard_rbf(X, X, lengthscales, kernel_var) + JITTER * torch.eye(N, dtype=X.dtype)
    L = torch.linalg.cholesky(K)
    u = pyro.sample("u", dist.Normal(X.new_zeros(N), X.new_ones(N)).to_event(1))
    f = mean + L @ u

    # --- likelihood
    with pyro.plate("data", N):
        pyro.sample("obs", dist.Bernoulli(logits=f), obs=y)
    return f


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def fit(X, y, num_warmup=500, num_samples=1000, num_chains=1,
        target_accept_prob=0.9, seed=0, linear_mean=False):
    """Run NUTS. Returns the fitted MCMC object.

    Production guidance: use >= 2 chains and check R-hat (~1.0) and the
    divergence count via mcmc.summary() / mcmc.diagnostics(). 500/1000 here is
    a reasonable default for ~200 wells; bump warmup if R-hat is off.

    linear_mean (opt-in): use an affine latent mean so off-support predictions
    trend toward 0/1 instead of reverting to the base rate (see gpc_model). The
    predictors auto-detect it from the sample sites (mean_slope present).
    """
    pyro.set_rng_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    base_rate = float(y.mean())

    kernel = NUTS(gpc_model, target_accept_prob=target_accept_prob,
                  max_tree_depth=10, jit_compile=False)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_warmup,
                num_chains=num_chains)
    mcmc.run(X, y, base_rate=base_rate, linear_mean=linear_mean)
    return mcmc


# ---------------------------------------------------------------------------
# Prediction: posterior predictive P(success) at new prospects
# ---------------------------------------------------------------------------
def _mean_at(samples, s, X_new, fallback):
    """Latent mean at X_new for posterior draw s. Affine (mean_intercept + X@mean_slope)
    if the linear-mean model was fit, else the constant 'mean', else the base-rate
    fallback. Returns a scalar or (M,) tensor; broadcasts against the GP conditional
    term. The stored whitened u already absorbs the training-mean residual, so only
    the NEW-point mean changes between the constant and linear-mean variants."""
    if "mean_slope" in samples:
        b0 = samples["mean_intercept"][s].to(torch.float64)
        b1 = samples["mean_slope"][s].to(torch.float64)
        return b0 + X_new @ b1
    if "mean" in samples:
        return samples["mean"][s].to(torch.float64)
    return fallback


@torch.no_grad()
def predict_proba(samples, X_train, X_new, base_rate, n_draws=1):
    """Monte-Carlo samples of P(success) at each row of X_new.

    For every posterior draw we form the GP conditional latent at X_new and
    push it through the sigmoid link. Returns shape (S * n_draws, M).

    Because we kept the whitened u, the usual conditional simplifies:
        f*_mean = mean + K*N L^{-T} u
        f*_var  = diag(K**) - || L^{-1} K*N^T ||^2
    so no second large solve against (f_train - mean) is needed.
    """
    X_train = torch.as_tensor(X_train, dtype=torch.float64)
    X_new = torch.as_tensor(X_new, dtype=torch.float64)
    if X_new.dim() == 1:
        X_new = X_new.unsqueeze(0)

    mean_fallback = math.log(base_rate / (1.0 - base_rate))
    S = samples["u"].shape[0]
    N = X_train.shape[0]
    M = X_new.shape[0]
    eye = JITTER * torch.eye(N, dtype=torch.float64)

    out = []
    for s in range(S):
        ls = samples["lengthscales"][s].to(torch.float64)
        var = samples["kernel_var"][s].to(torch.float64)
        u = samples["u"][s].to(torch.float64)
        m = _mean_at(samples, s, X_new, mean_fallback)  # scalar or affine (M,)

        K = ard_rbf(X_train, X_train, ls, var) + eye
        L = torch.linalg.cholesky(K)
        Ks = ard_rbf(X_new, X_train, ls, var)          # (M, N)
        kss = var.expand(M)                            # diag of K(X_new, X_new)

        tmp = torch.linalg.solve_triangular(L.t(), u.unsqueeze(-1), upper=True)
        f_mean = m + (Ks @ tmp).squeeze(-1)            # (M,)

        v = torch.linalg.solve_triangular(L, Ks.t(), upper=False)   # (N, M)
        f_var = (kss - (v ** 2).sum(0)).clamp(min=1e-9)             # (M,)

        eps = torch.randn(n_draws, M, dtype=torch.float64)
        f_draw = f_mean + eps * f_var.sqrt()
        out.append(torch.sigmoid(f_draw))

    return torch.cat(out, dim=0)


@torch.no_grad()
def predict_latent(samples, X_train, X_new, base_rate):
    """Posterior mean and std of the GP LATENT f* (logit scale) at each X_new row.

    Where predict_proba squashes f* through the sigmoid, this returns the *pre-link*
    moments -- the clean epistemic uncertainty used for risk weighting (the sigmoid
    is location-confounded, so a probability-scale std is a poor reliability signal).

    The full latent posterior is a mixture over draws s of N(f_mean[s], f_var[s]).
    By the law of total variance:
        mu_f  = mean_s f_mean[s]
        var_f = mean_s f_var[s]            (within-draw GP conditional variance)
              + var_s  f_mean[s]           (between-draw spread of the mean)
    The within-draw term -> kernel_var on an extrapolation (no analog reduces it),
    so sigma_f is large exactly where there is no analog support. Returns (mu_f,
    sigma_f), each shape (M,), both float64 on the logit scale.
    """
    X_train = torch.as_tensor(X_train, dtype=torch.float64)
    X_new = torch.as_tensor(X_new, dtype=torch.float64)
    if X_new.dim() == 1:
        X_new = X_new.unsqueeze(0)

    mean_fallback = math.log(base_rate / (1.0 - base_rate))
    S = samples["u"].shape[0]
    N = X_train.shape[0]
    M = X_new.shape[0]
    eye = JITTER * torch.eye(N, dtype=torch.float64)

    f_means = torch.empty(S, M, dtype=torch.float64)
    f_vars = torch.empty(S, M, dtype=torch.float64)
    for s in range(S):
        ls = samples["lengthscales"][s].to(torch.float64)
        var = samples["kernel_var"][s].to(torch.float64)
        u = samples["u"][s].to(torch.float64)
        m = _mean_at(samples, s, X_new, mean_fallback)  # scalar or affine (M,)

        K = ard_rbf(X_train, X_train, ls, var) + eye
        L = torch.linalg.cholesky(K)
        Ks = ard_rbf(X_new, X_train, ls, var)          # (M, N)
        kss = var.expand(M)                            # diag of K(X_new, X_new)

        tmp = torch.linalg.solve_triangular(L.t(), u.unsqueeze(-1), upper=True)
        f_means[s] = m + (Ks @ tmp).squeeze(-1)        # (M,)

        v = torch.linalg.solve_triangular(L, Ks.t(), upper=False)   # (N, M)
        f_vars[s] = (kss - (v ** 2).sum(0)).clamp(min=1e-9)         # (M,)

    mu_f = f_means.mean(0)
    var_f = f_vars.mean(0) + f_means.var(0, unbiased=False)         # total variance
    return mu_f, var_f.sqrt()


def summarize(prob_samples, lo=0.05, hi=0.95):
    """Collapse predictive samples -> point probability + credible band."""
    return {
        "prob": prob_samples.mean(0),
        "lo": prob_samples.quantile(lo, dim=0),
        "hi": prob_samples.quantile(hi, dim=0),
        "std": prob_samples.std(0),
    }


# ---------------------------------------------------------------------------
# Calibration / evaluation
# ---------------------------------------------------------------------------
def calibration_metrics(p, y):
    """Brier score and log loss -- the right graders for imbalanced probs.
    (Accuracy is misleading here; grade the probability, not a 0.5 cutoff.)"""
    p = torch.as_tensor(p, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    y = torch.as_tensor(y, dtype=torch.float64)
    brier = ((p - y) ** 2).mean()
    log_loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
    return {"brier": float(brier), "log_loss": float(log_loss)}


def reliability_table(p, y, n_bins=10):
    """Rows of (bin_centre, mean_predicted, observed_freq, count).
    A well-calibrated model has mean_predicted ~ observed_freq in every bin."""
    p = torch.as_tensor(p, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    rows = []
    for b in range(n_bins):
        a, c = edges[b], edges[b + 1]
        mask = (p >= a) & (p < c) if b < n_bins - 1 else (p >= a) & (p <= c)
        if int(mask.sum()) == 0:
            continue
        rows.append((float((a + c) / 2), float(p[mask].mean()),
                     float(y[mask].mean()), int(mask.sum())))
    return rows


def expected_calibration_error(p, y, n_bins=10):
    """ECE: average gap |mean_predicted - observed_freq|, weighted by bin count.
    A single number summarising the reliability diagram; 0 = perfectly calibrated."""
    rows = reliability_table(p, y, n_bins)
    n = len(p)
    return sum((cnt / n) * abs(mp - obs) for _, mp, obs, cnt in rows)


def roc_auc(p, y):
    """Area under the ROC curve via the Mann-Whitney rank-sum identity.

    AUC = P(score of a random success > score of a random dry hole). Ties in
    the score get averaged ranks, so repeated rating-vectors are handled
    correctly. Returns nan if one class is absent.
    """
    p = torch.as_tensor(p, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(p)
    sp = p[order]
    ranks = torch.arange(1, len(p) + 1, dtype=torch.float64)
    i, N = 0, len(p)                       # average ranks within tied groups
    while i < N:
        j = i
        while j + 1 < N and sp[j + 1] == sp[i]:
            j += 1
        ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    rank_full = torch.empty(N, dtype=torch.float64)
    rank_full[order] = ranks
    sum_pos = rank_full[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(p, y):
    """Average precision (area under the precision-recall curve).

    More informative than ROC-AUC when the positive class is the thing you care
    about and is not the majority. Computed as sum_k (R_k - R_{k-1}) * P_k.
    """
    p = torch.as_tensor(p, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = torch.argsort(p, descending=True)
    ys = y[order]
    tp = torch.cumsum(ys, 0)
    fp = torch.cumsum(1 - ys, 0)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = torch.cat([torch.zeros(1, dtype=torch.float64), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def discrimination_metrics(p, y):
    """ROC-AUC + PR-AUC together."""
    return {"roc_auc": roc_auc(p, y), "pr_auc": pr_auc(p, y)}


def reliability_plot(p, y, n_bins=10, path="reliability.png"):
    """Save a reliability diagram (needs matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = reliability_table(p, y, n_bins)
    pred = [r[1] for r in rows]
    obs = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(pred, obs, "o-", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed success frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def stratified_kfold_oof(X, y, k=5, seed=0, n_draws=1, fit_kwargs=None,
                         transform=None):
    """Out-of-fold predicted probabilities, stratified on the binary label.

    Refits NUTS k times -- this is the honest (not in-sample) calibration
    estimate, but it is k * (one fit) of compute. For a cheaper single-fit
    alternative, compute pointwise log-likelihoods and use PSIS-LOO (ArviZ).

    `transform` is an optional *supervised* feature encoder factory, called as
    `transform(X_tr, y_tr) -> enc` with `enc.transform(X) -> X'`. It is fit on the
    TRAIN fold only and applied to both slices INSIDE the loop, so a label-derived
    encoding (e.g. WoEEncoder) cannot leak the held-out labels. Pass the raw
    feature columns as `X` when using a transform. None -> X used as-is.
    """
    fit_kwargs = fit_kwargs or {}
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)

    folds = torch.full((len(y),), -1, dtype=torch.long)
    for cls in (0.0, 1.0):                       # stratify
        idx = torch.where(y == cls)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        for j, i in enumerate(idx):
            folds[i] = j % k

    oof = torch.zeros(len(y), dtype=torch.float64)
    for fold in range(k):
        te = folds == fold
        tr = ~te
        if transform is not None:                # fit encoder on TRAIN ONLY
            enc = transform(X[tr], y[tr])
            Xtr, Xte = enc.transform(X[tr]), enc.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        mcmc = fit(Xtr, y[tr], **fit_kwargs)
        br = float(y[tr].mean())
        ps = predict_proba(mcmc.get_samples(), Xtr, Xte, br, n_draws=n_draws)
        oof[te] = ps.mean(0)
    return oof


def repeated_kfold_oof(X, y, k=5, n_repeats=3, seed=0, n_draws=1, fit_kwargs=None,
                       transform=None):
    """Stack `n_repeats` independent stratified-k-fold OOF runs.

    Each repeat reshuffles the fold assignment (seed + r), so every repeat gives
    one full vector of out-of-fold probabilities over all N wells. Returns shape
    (n_repeats, N). Computing a metric per repeat and reporting mean +/- std
    across repeats is what turns a single noisy number -- the real risk at
    n~200 -- into an estimate with a spread you can trust.

    `transform` (a per-fold supervised encoder factory) is threaded down to every
    fold of every repeat; see `stratified_kfold_oof`.
    """
    runs = [stratified_kfold_oof(X, y, k=k, seed=seed + r, n_draws=n_draws,
                                 fit_kwargs=fit_kwargs, transform=transform)
            for r in range(n_repeats)]
    return torch.stack(runs)


def subset_features(X, names, keep):
    """Select feature columns by name. Returns (X[:, idx], list(keep))."""
    idx = [names.index(c) for c in keep]
    return X[:, idx], list(keep)


def _isotonic_increasing(values, weights):
    """Weighted pool-adjacent-violators -> non-decreasing fit. Tensors in/out.

    Used to enforce monotone WoE in ascending (worst->best) level order: it pools
    adjacent levels that violate monotonicity (e.g. an L4>L5 sampling reversal, or
    an L1>L2 wiggle that EB shrinkage itself can introduce because small cells
    shrink more), weighting each level by its count.
    """
    v = [float(x) for x in values]
    w = [float(x) for x in weights]
    blocks = [[v[i], w[i], i, i] for i in range(len(v))]   # [value, weight, lo, hi]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:        # violation -> pool
            tw = blocks[i][1] + blocks[i + 1][1]
            tv = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / tw
            blocks[i:i + 2] = [[tv, tw, blocks[i][2], blocks[i + 1][3]]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = [0.0] * len(v)
    for bv, _bw, lo, hi in blocks:
        for j in range(lo, hi + 1):
            out[j] = bv
    return torch.tensor(out, dtype=torch.float64)


class WoEEncoder:
    """Leakage-safe Weight-of-Evidence encoder for ordinal/discrete features.

    Fit on (X_raw, y) from a TRAINING fold only, then applied to held-out rows.
    Each column's levels are mapped to a base-rate-centred, empirical-Bayes-shrunk,
    isotonic Weight of Evidence:

        WoE(l) = logit P(y=1 | x=l) - logit(p0)      (0 = base rate = no evidence)

    so the encoded value is the evidence a level adds to the prior log-odds, on the
    same logit scale as the GP `mean` prior. This re-spaces ordinal levels by their
    empirical log-odds instead of assuming the integer codes 1..5 are evenly spaced.
    See FINDINGS.md for the full rationale and the calculation decisions.

    Used as a per-fold factory by `stratified_kfold_oof(..., transform=enc)`:
    calling the instance, `enc(X_tr, y_tr)`, returns a freshly fitted copy.
    """

    def __init__(self, combine="mean", monotone=True, ordinal_extrap="zero"):
        self.combine = combine        # "mean" -> (N,1) composite; None -> (N,D) WoE cols
        self.monotone = monotone      # isotonic (PAVA) in ascending level order
        # how to map a level OUTSIDE the fitted ordinal range (e.g. a "no anomaly"
        # level 0 below the worst observed 1): "zero" = WoE 0 (default, = the original
        # unseen->no-evidence behaviour, bitwise unchanged); "clamp" = nearest observed
        # level's WoE ("at least as bad as the worst"); "linear" = extend the end slope
        # (assert the monotone trend continues -- pairs with linear_mean). An *interior*
        # unseen level is always 0 (genuine no-evidence), regardless of this setting.
        self.ordinal_extrap = ordinal_extrap
        self.maps_ = None             # per-column {level_value: woe}
        self.b0_ = None               # train-fold base logit

    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        p0 = float(y.mean())
        self.b0_ = math.log(p0 / (1.0 - p0))
        self.maps_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            levels = torch.unique(col)                       # sorted ascending
            n = torch.tensor([float((col == L).sum()) for L in levels])
            k = torch.tensor([float(y[col == L].sum()) for L in levels])
            # empirical centred logit (WoE) with Haldane-Anscombe +0.5 correction
            woe = torch.log((k + 0.5) / (n - k + 0.5)) - self.b0_
            se2 = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)      # sampling var of WoE
            # empirical-Bayes shrink toward 0 (no evidence). tau^2 = between-level
            # dispersion beyond sampling noise (precision-weighted method of moments);
            # small/noisy cells (large se2) collapse to ~0, large cells keep signal.
            w = 1.0 / se2
            wbar = (woe * w).sum() / w.sum()
            tau2 = (w * (woe - wbar) ** 2).sum() / w.sum() - se2.mean()
            tau2 = float(tau2.clamp(min=0.0))
            woe = woe * (tau2 / (tau2 + se2))
            if self.monotone:                                # PAVA last (see helper)
                woe = _isotonic_increasing(woe, n)
            self.maps_.append({float(L): float(v) for L, v in zip(levels, woe)})
        return self

    def transform(self, X):
        X = torch.as_tensor(X, dtype=torch.float64)
        out = torch.zeros(X.shape[0], len(self.maps_), dtype=torch.float64)
        for j, m in enumerate(self.maps_):
            levels = sorted(m)
            lo, hi = levels[0], levels[-1]
            slo = (m[levels[1]] - m[lo]) if len(levels) > 1 else 0.0     # low-end slope
            shi = (m[hi] - m[levels[-2]]) if len(levels) > 1 else 0.0    # high-end slope
            col = X[:, j]
            for i in range(X.shape[0]):
                v = float(col[i])
                if v in m:
                    out[i, j] = m[v]
                elif (v < lo or v > hi) and self.ordinal_extrap != "zero":
                    if self.ordinal_extrap == "clamp":                  # nearest observed
                        out[i, j] = m[lo] if v < lo else m[hi]
                    else:                                               # "linear": extend slope
                        out[i, j] = (m[lo] + slo * (v - lo)) if v < lo else (m[hi] + shi * (v - hi))
                else:
                    out[i, j] = 0.0          # interior-unseen / unknown / "zero" -> no evidence
        if self.combine == "mean":
            return out.mean(dim=1, keepdim=True)
        return out

    def __call__(self, X, y):        # factory form for the CV transform= hook
        return WoEEncoder(self.combine, self.monotone, self.ordinal_extrap).fit(X, y)


class WoEMeanPlusRaw:
    """WoE-encode the first `n_woe` columns into a single mean composite, then
    concatenate the remaining columns unchanged.
    """

    def __init__(self, n_woe, monotone=True, ordinal_extrap="zero"):
        self.n_woe = n_woe
        self.monotone = monotone
        self.ordinal_extrap = ordinal_extrap     # see WoEEncoder; "linear" => no-anomaly path
        self.woe_ = None

    def fit(self, X, y):
        X = torch.as_tensor(X, dtype=torch.float64)
        self.woe_ = WoEEncoder(combine="mean", monotone=self.monotone,
                               ordinal_extrap=self.ordinal_extrap).fit(X[:, :self.n_woe], y)
        return self

    def transform(self, X):
        X = torch.as_tensor(X, dtype=torch.float64)
        woe = self.woe_.transform(X[:, :self.n_woe])        # (N, 1)
        return torch.cat([woe, X[:, self.n_woe:]], dim=1)   # + raw passthrough cols

    def __call__(self, X, y):        # factory form for the CV transform= hook
        return WoEMeanPlusRaw(self.n_woe, self.monotone, self.ordinal_extrap).fit(X, y)

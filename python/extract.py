"""MATLAB → Python port of event_detection/main.m + helpers.

Mirrors the math in:
  - main.m            (window/overlap/posterior threshold)
  - Events_Features_Extraction.m  (7-D feature)
  - GMM_EM.m          (custom EM with random init + reg)
  - Event_Extract.m   (1500-sample footstep extraction with Tukey window)

MATLAB-isms replicated explicitly: see matlab_smooth, matlab_kurtosis,
matlab_q25, gausswin.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.io
from scipy import signal as ss
from scipy import stats
from scipy.stats import multivariate_normal


# ---------- MATLAB primitive replications ----------

def matlab_smooth(x: np.ndarray, span: int = 5) -> np.ndarray:
    """MATLAB smooth(x, span) with default 'moving' method.

    Endpoints use a shrinking symmetric window: y[i] uses
    span_i = min(span, 2*i+1, 2*(n-1-i)+1).

    Vectorized for the inner region; only the small endpoint zone is
    looped, so this stays O(n) in time but constant in Python overhead.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    half = span // 2
    y = np.empty(n, dtype=np.float64)

    if n < span:
        for i in range(n):
            h = min(i, n - 1 - i, half)
            y[i] = x[i - h : i + h + 1].mean()
        return y

    kernel = np.ones(span, dtype=np.float64) / span
    y[half : n - half] = np.convolve(x, kernel, mode="valid")
    for i in range(half):
        y[i] = x[: 2 * i + 1].mean()
        y[n - 1 - i] = x[n - 2 * i - 1 :].mean()
    return y


def gausswin(N: int, alpha: float) -> np.ndarray:
    """MATLAB gausswin(N, alpha): w(n) = exp(-0.5 * (alpha * n / (N/2))^2)
    where n is centered at (N-1)/2.
    """
    n = np.arange(N) - (N - 1) / 2
    return np.exp(-0.5 * (alpha * n / ((N - 1) / 2)) ** 2)


def matlab_kurtosis(x: np.ndarray) -> float:
    """MATLAB kurtosis(x) — non-Fisher (no -3), biased (population)."""
    return float(stats.kurtosis(x, fisher=False, bias=True))


def matlab_q25(x: np.ndarray) -> float:
    """MATLAB quantile(x, 0.25) — Hyndman-Fan type 5 (Hazen plotting position).

    numpy >= 1.22: np.quantile(method="hazen"). Older numpy doesn't have type 5
    in 'interpolation' enum, so fall back to manual Hazen.
    """
    try:
        return float(np.quantile(x, 0.25, method="hazen"))
    except TypeError:
        s = np.sort(x)
        N = len(s)
        # Hazen: position h = p*N + 0.5 (1-based)
        h = 0.25 * N + 0.5
        if h <= 1:
            return float(s[0])
        if h >= N:
            return float(s[-1])
        lo = int(np.floor(h)) - 1
        hi = lo + 1
        frac = h - np.floor(h)
        return float(s[lo] * (1 - frac) + s[hi] * frac)


def nextpow2(L: int) -> int:
    return int(np.ceil(np.log2(L)))


# ---------- Feature extraction ----------

def extract_features(sig: np.ndarray, fs: int = 8000) -> np.ndarray:
    """7-D feature vector from a single Gaussian-windowed segment.

    Order: std, kurtosis, rms, q25, ||FFT||² in 40-80, 80-120, 120-160 Hz.
    """
    sig = sig.ravel()
    feat = np.empty(7)
    feat[0] = float(np.std(sig, ddof=1))           # MATLAB std defaults to N-1
    feat[1] = matlab_kurtosis(sig)
    feat[2] = float(np.sqrt(np.mean(sig ** 2)))    # rms
    feat[3] = matlab_q25(sig)

    L = len(sig)
    NFFT = 8 * (1 << nextpow2(L))
    F = np.fft.fft(sig, NFFT) / L
    F = 2 * np.abs(F[: NFFT // 2 + 1])
    f = (fs / 2.0) * np.linspace(0.0, 1.0, NFFT // 2 + 1)

    step = 40
    for i, j in enumerate([40, 80, 120]):
        # MATLAB: sum(f<=j)+1 : sum(f<=j+step)  (inclusive both, 1-based)
        # Python: f<=j gives count → 0-based start; f<=(j+step) gives 0-based end (exclusive)
        lo = int((f <= j).sum())
        hi = int((f <= j + step).sum())
        feat[4 + i] = float(np.sum(F[lo:hi] ** 2))

    return feat


def slide_features(geo_data: np.ndarray, fs: int = 8000,
                   window_sec: float = 0.35, overlap: float = 0.40,
                   tau: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows over geo_data, return (features [N×7], indices [N×2]).

    indices[:,0]=start (0-based, inclusive), indices[:,1]=stop (0-based, exclusive).
    Number of segments matches MATLAB main.m's loop:
      while iter < length(geo_data): take floor(1 + (L-W) / floor((1-ovrlap)*W)) full,
      plus one partial if next start is still inside signal.
    """
    geo_data = geo_data.ravel().astype(np.float64)
    L = len(geo_data)
    W = int(window_sec * fs)              # 2800 for 0.35*8000
    step = int(np.floor((1 - overlap) * W))   # 1680

    n_full = int(np.floor(1 + (L - W) / step))

    weight_full = gausswin(W, tau)

    feats, idx = [], []
    for i in range(n_full):
        start = i * step
        stop = start + W
        if stop > L:
            stop = L
        seg = geo_data[start:stop]
        w = weight_full if len(seg) == W else gausswin(len(seg), tau)
        feats.append(extract_features(w * seg, fs))
        idx.append((start, stop))

    return np.asarray(feats), np.asarray(idx)


# ---------- GMM (EM) ----------

@dataclass
class GMMParams:
    mu: np.ndarray   # (K, D)
    cov: np.ndarray  # (K, D, D)
    phi: np.ndarray  # (K,)
    iters: int
    converged: bool


def gmm_em(X: np.ndarray, K: int = 2, max_iter: int = 1000, tol: float = 1e-12,
           reg: float = 1e-4, seed: Optional[int] = None) -> GMMParams:
    """Replicate event_detection/GMM_EM.m.

    - Random row indices for μ init (randperm)
    - Both clusters' Σ initialized to full data cov
    - Σ regularization: + reg * I (MATLAB uses 0.0001)
    - Convergence: |‖μ_new‖₂ - ‖μ_prev‖₂| < tol  (MATLAB uses operator 2-norm)
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    rand_idx = rng.choice(N, K, replace=False)

    mu = X[rand_idx].copy()
    base_cov = np.cov(X, rowvar=False)
    cov = np.broadcast_to(base_cov, (K, D, D)).copy()
    phi = np.full(K, 1.0 / K)

    prev_norm = np.inf
    converged = False
    for it in range(max_iter):
        # E-step
        log_w = np.empty((N, K))
        for k in range(K):
            log_w[:, k] = (np.log(phi[k] + 1e-300)
                           + multivariate_normal.logpdf(X, mu[k], cov[k],
                                                        allow_singular=True))
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        s = w.sum(axis=1, keepdims=True)
        w = np.where(s > 0, w / s, 0.0)
        w = np.nan_to_num(w, nan=0.0)

        # M-step
        Nk = w.sum(axis=0)
        phi = Nk / N
        for k in range(K):
            mu[k] = (w[:, k:k+1] * X).sum(axis=0) / Nk[k]
            diff = X - mu[k]
            cov[k] = (w[:, k:k+1] * diff).T @ diff / Nk[k] + reg * np.eye(D)

        new_norm = float(np.linalg.norm(mu, ord=2))
        err = abs(new_norm - prev_norm)
        prev_norm = new_norm
        if err < tol:
            converged = True
            break

    return GMMParams(mu=mu, cov=cov, phi=phi, iters=it + 1, converged=converged)


# ---------- Posterior cluster assignment + threshold ----------

def assign_clusters(feats: np.ndarray, gmm: GMMParams,
                    threshold: float = 0.90) -> np.ndarray:
    """For each window, compute posterior, argmax, map to event/noise label.

    Cluster with larger |Σ| ⇒ event class (label 1); other ⇒ noise (label 0).
    If max posterior < threshold, override to noise.
    """
    dets = np.array([np.linalg.det(gmm.cov[k]) for k in range(gmm.mu.shape[0])])
    event_cluster = int(np.argmax(dets))
    return assign_clusters_frozen(feats, gmm.mu, gmm.cov, gmm.phi,
                                   event_cluster, threshold)


def assign_clusters_frozen(feats: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                           phi: np.ndarray, event_cluster: int,
                           threshold: float = 0.90) -> np.ndarray:
    """Inference-time variant: caller supplies a pre-determined event_cluster.

    Skips det() so the .npz-loaded params are the single source of truth.
    """
    K = mu.shape[0]
    p = np.zeros((len(feats), K))
    for k in range(K):
        p[:, k] = multivariate_normal.pdf(feats, mu[k], cov[k],
                                          allow_singular=True) * phi[k]
    s = p.sum(axis=1, keepdims=True)
    p_norm = p / np.where(s > 0, s, 1.0)
    max_p = p_norm.max(axis=1)
    argmax = p_norm.argmax(axis=1)
    label = np.where(argmax == event_cluster, 1, 0)
    label[max_p < threshold] = 0
    return label


def load_gmm_npz(path: str) -> dict:
    """Load frozen GMM params exported by export_gmm.py.

    Returns dict with keys: mu, cov, phi, event_cluster (int), threshold (float).
    """
    z = np.load(path)
    return dict(
        mu=z["mu"],
        cov=z["cov"],
        phi=z["phi"],
        event_cluster=int(z["event_cluster"]),
        threshold=float(z["threshold"]),
    )


def apply_frozen_gmm(geo_data: np.ndarray, params: dict, fs: int = 8000) -> np.ndarray:
    """Inference path: smooth → features → frozen GMM classify → footsteps."""
    g = matlab_smooth(geo_data, span=5)
    feats, indices = slide_features(g, fs=fs)
    labels = assign_clusters_frozen(
        feats, params["mu"], params["cov"], params["phi"],
        params["event_cluster"], params["threshold"],
    )
    return extract_footsteps(g, indices, labels)


# ---------- Hilbert-envelope detector (replacement for frozen GMM) ----------

def apply_envelope_detector(
    geo_data: np.ndarray,
    fs: int = 8000,
    k_mad: float = 8.0,
    env_smooth_ms: float = 30.0,
    min_sep_ms: float = 120.0,
    align_radius_ms: float = 30.0,
    footfall_len: int = 1500,
    tukey_alpha: float = 0.5,
    peak_offset: int = 400,
) -> np.ndarray:
    """Hilbert-envelope based footstep detection. Drop-in replacement for
    apply_frozen_gmm; returns the same (M, 1500) Tukey-windowed crops.

    Algorithm:
      1. matlab_smooth (span=5)              # same as GMM path
      2. |analytic signal|                   # Hilbert envelope
      3. moving-average smooth (env_smooth_ms)
      4. adaptive threshold = median + k_mad * MAD * 1.4826   # robust σ
      5. scipy.signal.find_peaks(height=thr, distance=min_sep_ms)
      6. snap each peak to argmax(|signal|) within ±align_radius_ms
         (matches Event_Extract.m's inner peak refinement)
      7. extract 1500-sample slice centered at peak-400, Tukey window,
         zero-pad to footfall_len.

    Defaults validated on P14 (84.89% v2-classifier accuracy vs 83.38% GMM)
    and on our 30 s STM32 capture (25 detections vs 10, no quiet-region FPs).
    """
    g = matlab_smooth(geo_data, span=5).ravel()
    L = len(g)

    env = np.abs(ss.hilbert(g))
    n_smooth = max(1, int(env_smooth_ms * 1e-3 * fs))
    if n_smooth > 1:
        kern = np.ones(n_smooth) / n_smooth
        env = np.convolve(env, kern, mode="same")

    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    thr = med + k_mad * mad * 1.4826

    min_dist = max(1, int(min_sep_ms * 1e-3 * fs))
    env_peaks, _ = ss.find_peaks(env, height=thr, distance=min_dist)

    if len(env_peaks) == 0:
        return np.zeros((0, footfall_len))

    radius = max(1, int(align_radius_ms * 1e-3 * fs))
    full_window = ss.windows.tukey(footfall_len, tukey_alpha)
    abs_g = np.abs(g)

    out = []
    for p in env_peaks:
        lo = max(0, int(p) - radius)
        hi = min(L, int(p) + radius)
        mn = int(np.argmax(abs_g[lo:hi])) + lo

        strt = mn - peak_offset
        stop = strt + footfall_len
        wnd = full_window.copy()
        if strt < 0:
            wnd = wnd[: stop - 0]
            strt = 0
        if stop > L:
            wnd = wnd[: L - strt]
            stop = L

        seg = g[strt:stop]
        windowed = wnd * seg
        row = np.zeros(footfall_len)
        row[: len(windowed)] = windowed
        out.append(row)

    return np.asarray(out)


# ---------- Footstep extraction (Event_Extract.m) ----------

def extract_footsteps(signal_arr: np.ndarray, indices: np.ndarray,
                      labels: np.ndarray, footfall_len: int = 1500,
                      tukey_alpha: float = 0.5,
                      peak_offset: int = 400) -> np.ndarray:
    """Replicate event_detection/Event_Extract.m.

    Group consecutive event windows (label==1), find peak |signal| in
    [start_first, stop_last], take 1500-sample slice centered at peak-400,
    apply Tukey window (truncate at edges), zero-pad to 1500.

    Returns array (M, footfall_len).
    """
    signal_arr = signal_arr.ravel()
    L = len(signal_arr)
    event_idx = np.flatnonzero(labels == 1)
    if len(event_idx) == 0:
        return np.zeros((0, footfall_len))

    full_window = ss.windows.tukey(footfall_len, tukey_alpha)

    out = []
    j = 0
    n = len(event_idx)
    # MATLAB: while j < length(...). Last index n-1 (0-based) is excluded if standalone.
    while j < n - 1:
        run_start = j
        # Walk forward while consecutive
        while j < n - 1 and event_idx[j + 1] == event_idx[j] + 1:
            j += 1
        run_end = j  # last index in run

        evnt_start = int(indices[event_idx[run_start], 0])
        evnt_stop = int(indices[event_idx[run_end], 1])

        # Find peak
        seg = signal_arr[evnt_start:evnt_stop]
        if len(seg) == 0:
            j += 1
            continue
        peak_local = int(np.argmax(np.abs(seg)))
        # MATLAB: mn = mn + Evnt_Start (off-by-one quirk replicated)
        mn = peak_local + evnt_start

        strt = mn - peak_offset
        stop = strt + footfall_len  # exclusive
        wnd = full_window.copy()
        if strt < 0:
            wnd = wnd[: stop - 0]
            strt = 0
        if stop > L:
            wnd = wnd[: L - strt]
            stop = L

        seg = signal_arr[strt:stop]
        windowed = wnd * seg
        # zero-pad to 1500
        out_row = np.zeros(footfall_len)
        out_row[: len(windowed)] = windowed
        out.append(out_row)

        j += 1

    return np.asarray(out)


# ---------- High-level convenience ----------

def load_person_raw(person: str, data_root: str = "E:/Terra/data/raw",
                    n_files: int = 4) -> np.ndarray:
    """Load Pn_1..Pn_n_files .mat files, vstack like NewDatasetCreation2.m."""
    parts = []
    for i in range(1, n_files + 1):
        p = f"{data_root}/{person}/{person}_{i}.mat"
        d = scipy.io.loadmat(p)
        parts.append(d["geo_data"].ravel())
    return np.concatenate(parts).astype(np.float64)


def train_gmm_on_first_seconds(geo_data: np.ndarray, n_seconds: int = 100,
                                fs: int = 8000, seed: Optional[int] = 0) -> Tuple[GMMParams, np.ndarray]:
    """main.m initial block: take first n_seconds, smooth, slide+features, GMM.

    Returns (GMMParams, training_features).
    """
    n = n_seconds * fs
    g = matlab_smooth(geo_data[:n], span=5)
    feats, _ = slide_features(g, fs=fs)
    gmm = gmm_em(feats, K=2, seed=seed)
    return gmm, feats


def apply_to_person(geo_data: np.ndarray, gmm: GMMParams, fs: int = 8000,
                    threshold: float = 0.90) -> np.ndarray:
    """For a full person signal: smooth → features → posterior+threshold → footsteps."""
    g = matlab_smooth(geo_data, span=5)
    feats, indices = slide_features(g, fs=fs)
    labels = assign_clusters(feats, gmm, threshold=threshold)
    return extract_footsteps(g, indices, labels)

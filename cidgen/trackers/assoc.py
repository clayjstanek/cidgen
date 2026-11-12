"""Per-look assignment marginals for soft association (JPDA and the variational tracker).

The joint assignment of one radar look: each report goes to at most one track or to clutter, each track
takes at most one report (or is missed). With W[i, j] = the log-odds score of "track i produced report j"
against "report j is clutter and track i was missed",

    q(A) ∝ 1{A valid} exp(sum_ij A_ij W_ij)

and the marginals P(A_ij = 1) are what JPDA and the VB tracker need. Two solvers:

  exact_marginals   exact, forward-backward over subsets of tracks: O(M N 2^N), used for N <= 12 tracks
                    (vectorised version of the subset DP in testVB_CVSystem_V3.m / vbtracker.assign)
  bp_marginals      loopy belief propagation for bipartite matching (Williams & Lau 2014), for larger
                    clusters; accurate in practice, exact on trees
Both return G (N+1, M): rows 0..N-1 tracks, row N clutter; columns sum to 1. Track i's miss probability is
1 - G[i].sum().
"""
from __future__ import annotations

import numpy as np


def logsumexp(a, axis=None, keepdims=False):
    """Plain numpy log-sum-exp (scipy's is far slower on the small arrays used here)."""
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out if keepdims else (np.squeeze(out, axis=axis) if axis is not None else out.item())

_MASKS = {}


def _mask_index(N):
    if N not in _MASKS:
        masks = np.arange(1 << N)
        without = [masks[(masks >> i) & 1 == 0] for i in range(N)]
        _MASKS[N] = without
    return _MASKS[N]


def exact_marginals(W):
    """Exact assignment marginals for W (N, M) of log-odds scores."""
    W = np.asarray(W, float)
    N, M = W.shape
    if M == 0:
        return np.zeros((N + 1, 0))
    if N == 0:
        return np.vstack([np.ones((1, M))])
    if N > 12:
        raise ValueError("exact_marginals: N > 12, use bp_marginals")
    if N == 1:                                                   # closed form
        e = np.exp(np.minimum(W[0], 700.0))
        g = e / (1.0 + e.sum())
        return np.vstack([g, 1.0 - g])
    without = _mask_index(N)
    nm = 1 << N
    Fw = np.full((M + 1, nm), -np.inf); Fw[0, 0] = 0.0
    for j in range(M):
        nxt = Fw[j].copy()                                       # report j -> clutter
        for i in range(N):
            src = without[i]
            nxt[src | (1 << i)] = np.logaddexp(nxt[src | (1 << i)], Fw[j, src] + W[i, j])
        Fw[j + 1] = nxt
    Bw = np.full((M + 1, nm), 0.0)
    for j in range(M - 1, -1, -1):
        cur = Bw[j + 1].copy()
        for i in range(N):
            src = without[i]
            cur[src] = np.logaddexp(cur[src], W[i, j] + Bw[j + 1, src | (1 << i)])
        Bw[j] = cur
    logZ = logsumexp(Fw[M])
    G = np.zeros((N + 1, M))
    G[N] = np.exp(logsumexp(Fw[:M] + Bw[1:], axis=1) - logZ)
    for i in range(N):
        src = without[i]
        G[i] = np.exp(logsumexp(Fw[:M, src] + Bw[1:, src | (1 << i)], axis=1) + W[i] - logZ)
    return G


def bp_marginals(W, n_iter=200, tol=1e-9):
    """Loopy BP for the same marginals (Williams & Lau), in log space."""
    W = np.asarray(W, float)
    N, M = W.shape
    if M == 0:
        return np.zeros((N + 1, 0))
    if N == 0:
        return np.ones((1, M))
    nu = np.zeros((N, M))                                        # log messages report j -> track i
    for _ in range(n_iter):
        a = W + nu                                               # track side
        # mu_ij = W_ij - log(1 + sum_{j' != j} exp(a_ij'))
        others = _log_sub(logsumexp(a, axis=1, keepdims=True), a)
        mu = W - np.logaddexp(0.0, others)
        oth_mu = _log_sub(logsumexp(mu, axis=0, keepdims=True), mu)
        new = -np.logaddexp(0.0, oth_mu)
        if np.max(np.abs(new - nu)) < tol:
            nu = new; break
        nu = new
    a = W + nu
    lz = np.logaddexp(0.0, logsumexp(a, axis=1, keepdims=True))
    P = np.exp(a - lz)                                           # track-side beliefs
    P = np.minimum(P, 1.0)
    col = P.sum(0)
    G = np.vstack([P, np.clip(1.0 - col, 0.0, 1.0)[None]])
    return G / G.sum(0, keepdims=True)


def _log_sub(log_total, log_part):
    """log(exp(log_total) - exp(log_part)), elementwise, safe when the difference is tiny."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = log_part - log_total
        out = log_total + np.log1p(-np.exp(np.minimum(d, -1e-300)))
    return np.where(np.isfinite(out), out, -np.inf)


def marginals(W, exact_max=12, n_iter=200):
    W = np.asarray(W, float)
    return exact_marginals(W) if W.shape[0] <= exact_max else bp_marginals(W, n_iter)


def enumerate_marginals(W):
    """Brute force (tests only): enumerate every valid assignment."""
    import itertools
    N, M = W.shape
    G = np.zeros((N + 1, M)); Z = 0.0
    for a in itertools.product(range(N + 1), repeat=M):          # a[j] = track or N (clutter)
        used = [x for x in a if x < N]
        if len(used) != len(set(used)):
            continue
        w = np.exp(sum(W[x, j] for j, x in enumerate(a) if x < N))
        Z += w
        for j, x in enumerate(a):
            G[x, j] += w
    return G / Z


# ---------------------------------------------------------------------- merged (unresolved) reports
def exact_set_marginals(events, N, M):
    """Exact marginals when a report may come from a SET of tracks (unresolved targets).

    events[j] = list of (mask, logw): report j explained by the tracks in bit-mask `mask` (one bit = an
    ordinary single-track assignment), with log-odds weight logw against "report j is clutter and those
    tracks were missed". Each track is used at most once per look. Returns (P, Pc): P[j] = array of event
    probabilities (same order as events[j]), Pc[j] = probability report j is clutter.
    Cost O(M 2^N E) for E events per report; for N <= 12.
    """
    nm = 1 << N
    allm = np.arange(nm)
    Fw = np.full((M + 1, nm), -np.inf); Fw[0, 0] = 0.0
    for j in range(M):
        nxt = Fw[j].copy()
        for mask, lw in events[j]:
            src = allm[(allm & mask) == 0]
            nxt[src | mask] = np.logaddexp(nxt[src | mask], Fw[j, src] + lw)
        Fw[j + 1] = nxt
    Bw = np.zeros((M + 1, nm))
    for j in range(M - 1, -1, -1):
        cur = Bw[j + 1].copy()
        for mask, lw in events[j]:
            src = allm[(allm & mask) == 0]
            cur[src] = np.logaddexp(cur[src], lw + Bw[j + 1, src | mask])
        Bw[j] = cur
    logZ = logsumexp(Fw[M])
    P, Pc = [], np.zeros(M)
    for j in range(M):
        Pc[j] = np.exp(logsumexp(Fw[j] + Bw[j + 1]) - logZ)
        pj = []
        for mask, lw in events[j]:
            src = allm[(allm & mask) == 0]
            pj.append(np.exp(logsumexp(Fw[j, src] + lw + Bw[j + 1, src | mask]) - logZ))
        P.append(np.array(pj))
    return P, Pc


def enumerate_set_marginals(events, N, M):
    """Brute force for tests: every choice of one event (or clutter) per report, tracks used once."""
    import itertools
    opts = [[(-1, 0.0)] + [(k, lw) for k, (_, lw) in enumerate(events[j])] for j in range(M)]
    P = [np.zeros(len(events[j])) for j in range(M)]; Pc = np.zeros(M); Z = 0.0
    for combo in itertools.product(*opts):
        used = 0; ok = True; w = 0.0
        for j, (k, lw) in enumerate(combo):
            if k >= 0:
                m = events[j][k][0]
                if used & m:
                    ok = False; break
                used |= m; w += lw
        if not ok:
            continue
        w = np.exp(w); Z += w
        for j, (k, _) in enumerate(combo):
            if k >= 0:
                P[j][k] += w
            else:
                Pc[j] += w
    return [p / Z for p in P], Pc / Z

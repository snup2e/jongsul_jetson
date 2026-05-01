"""Compute intra-class correlation ceiling per person from a1.mat.

If P1's typical within-class corr is ~0.93, then 0.93 is the actual ceiling
for any noisy extraction of P1, and our pipeline is fine.
"""
import numpy as np
import scipy.io

a1 = scipy.io.loadmat("E:/Terra/data/interim/a1.mat")["footstep_feat"]
labels = a1[:, -1].astype(int)

print(f"{'P':<5} {'N':>6} {'mean corr':>10} {'median':>9} {'p5':>7} {'p95':>7}")
print("-" * 55)
rng = np.random.default_rng(0)
for pid in [1, 14, 50, 100]:
    rows = a1[labels == pid][:, :-1]
    n = len(rows)
    # sample up to 200 rows for speed
    if n > 200:
        sel = rng.choice(n, 200, replace=False)
        sub = rows[sel]
    else:
        sub = rows
    Z = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-12)
    C = (Z @ Z.T) / Z.shape[1]
    # off-diagonal upper triangle
    iu = np.triu_indices_from(C, k=1)
    cs = C[iu]
    print(f"P{pid:<4d} {n:>6d} {cs.mean():>10.4f} {np.median(cs):>9.4f} "
          f"{np.percentile(cs, 5):>7.4f} {np.percentile(cs, 95):>7.4f}")

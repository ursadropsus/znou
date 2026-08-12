#!/usr/bin/env python3
"""
unreachable_certificate.py -- SPEC.md section 4.1.

For MLP layer l of GPT-2, bound each neuron's pre-activation over the
LayerNorm manifold and report any neuron that is provably never the argmax
at any position, for any input, under any of the four quadrants.

    python unreachable_certificate.py --layer 5
    python unreachable_certificate.py --selftest        # no model needed

Sound, not complete: an empty result proves nothing.
"""
import argparse
import numpy as np


def gelu_new(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_argmin():
    """Locate the interior minimum of gelu_new numerically."""
    grid = np.linspace(-3.0, 0.0, 300001)
    return float(grid[np.argmin(gelu_new(grid))])


def bounds(gamma, beta, W, b):
    """Exact bounds on pre-activation over {u : 1'u = 0, ||u|| <= sqrt(d)}.

    gamma, beta : (d,)        ln_2 weight and bias
    W           : (d, m)      c_fc weight
    b           : (m,)        c_fc bias
    returns L, U : (m,), (m,)
    """
    d = W.shape[0]
    V = gamma[:, None] * W                       # v_j = gamma * W[:, j]
    C = beta @ W + b                             # c_j
    PV = V - V.mean(axis=0, keepdims=True)       # project out all-ones
    radius = np.sqrt(d) * np.linalg.norm(PV, axis=0)
    return C - radius, C + radius


def certify(L, U):
    """Return (certified_indices, theta, g_hat, g_check)."""
    xstar = gelu_argmin()
    gL, gU = gelu_new(L), gelu_new(U)
    g_hat = np.maximum(gL, gU)                                  # max of g on [L, U]
    interior = (L <= xstar) & (xstar <= U)
    g_check = np.where(interior, gelu_new(xstar), np.minimum(gL, gU))
    theta = g_check.max()
    return np.flatnonzero(g_hat < theta), theta, g_hat, g_check


def selftest():
    rng = np.random.default_rng(0)
    d, m = 768, 3072
    gamma = rng.normal(1, 0.3, d)
    beta = rng.normal(0, 0.1, d)
    W = rng.normal(0, 0.05, (d, m))
    b = rng.normal(0, 0.05, m)

    L, U = bounds(gamma, beta, W, b)

    # random points on the manifold must respect the bounds
    X = rng.normal(size=(5000, d))
    X -= X.mean(1, keepdims=True)
    X *= np.sqrt(d) / np.linalg.norm(X, axis=1, keepdims=True)
    A = (X * gamma) @ W + b
    assert (A.max(0) <= U + 1e-6).all(), "upper bound violated"
    assert (A.min(0) >= L - 1e-6).all(), "lower bound violated"

    # bounds are attained exactly
    V = gamma[:, None] * W
    PV = V - V.mean(axis=0, keepdims=True)
    for j in (7, 100, 3000):
        u = np.sqrt(d) * PV[:, j] / np.linalg.norm(PV[:, j])
        got = ((gamma * u + beta) @ W + b)[j]
        assert abs(got - U[j]) < 1e-8 * max(1.0, abs(U[j])), (j, got, U[j])

    assert abs(gelu_argmin() - (-0.7517)) < 1e-2
    idx, theta, gh, gc = certify(L, U)
    print(f"selftest ok  (synthetic weights: {len(idx)} certified, theta={theta:.6f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    from transformers import GPT2LMHeadModel

    mdl = GPT2LMHeadModel.from_pretrained(args.model)
    layers = range(len(mdl.transformer.h)) if args.all_layers else [args.layer]

    for l in layers:
        blk = mdl.transformer.h[l]
        gamma = blk.ln_2.weight.detach().numpy().astype(np.float64)
        beta = blk.ln_2.bias.detach().numpy().astype(np.float64)
        W = blk.mlp.c_fc.weight.detach().numpy().astype(np.float64)   # (d, m)
        b = blk.mlp.c_fc.bias.detach().numpy().astype(np.float64)
        if W.shape[0] != gamma.shape[0]:
            W = W.T

        L, U = bounds(gamma, beta, W, b)
        idx, theta, g_hat, _ = certify(L, U)
        xstar = gelu_argmin()
        always_on = int((L > xstar).sum())

        print(f"layer {l}: d={W.shape[0]} m={W.shape[1]}")
        print(f"  pre-activation range: [{L.min():.3f}, {U.max():.3f}]")
        print(f"  Theta = {theta:.6f}   (highest guaranteed floor over all neurons)")
        print(f"  certified unreachable: {len(idx)}")
        if len(idx):
            print(f"  {idx.tolist()[:40]}{' ...' if len(idx) > 40 else ''}")
            return
        margin = float((g_hat - theta).min())
        print(f"  none. closest neuron sits {margin:+.6f} above Theta")
        if always_on == 0:
            print(f"  every neuron's range straddles the GELU minimum "
                  f"(x* = {xstar:.4f}), so every floor collapses to min(g).")
            print(f"  Theta == min(g) == {gelu_new(xstar):.6f}, and no g_hat can fall "
                  f"below the global minimum:")
            print(f"  the test is structurally incapable of firing at this layer, "
                  f"not merely empty.")
            print(f"  non-vacuity would need at least one neuron with L_j > x*; "
                  f"there are {always_on}.")


if __name__ == "__main__":
    main()
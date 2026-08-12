#!/usr/bin/env python3
"""
token_sweep.py -- exhaustive single-token census of J-Space.

Runs every token in the GPT-2 vocabulary (50,257) through layer L and records,
for each quadrant, which system it lands in. This is a COMPLETE result for
one-token strings, not an estimate: a system that appears here is reachable,
full stop.

Also emits:
  * the BOS probe -- the constant activation sitting at position 1 of every
    Implicit input, which competes against your text under Resonance
  * per-neuron maximum observed activation, a measured (not relaxed) proxy
    for how loud each neuron can get -- better ghost-test data than U_j

    python token_sweep.py
    python token_sweep.py --db znou_exchange.db     # + ghost test
    python token_sweep.py --limit 2000              # quick smoke test
    python token_sweep.py --selftest                # no model needed

Note: for a single-token EXPLICIT input there is only one position, so
Resonance and Inference are identical by construction. Three quadrants are
distinguishable here, not four.
"""
import argparse
import math
import sys

import numpy as np

BOS = 50256


# --------------------------------------------------------------------------
# pure helpers (testable without torch)
# --------------------------------------------------------------------------


def rank_sum(a, b):
    """Mann-Whitney U, normal approximation with tie correction.

    Returns (U1, z, p_two_sided). a, b are 1-D arrays.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")

    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    sortv = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)

    # average ranks within tie groups; accumulate tie correction
    tie_term = 0.0
    i = 0
    while i < len(sortv):
        j = i
        while j + 1 < len(sortv) and sortv[j + 1] == sortv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
            t = j - i + 1
            tie_term += t ** 3 - t
        i = j + 1

    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return U1, float("nan"), float("nan")
    z = (U1 - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return U1, z, p


def esc(s):
    """Make a BPE token safe for a TSV cell."""
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def summarise(winners, m):
    """winners: 1-D int array. Returns (reached_count, wins_per_neuron)."""
    wins = np.bincount(np.asarray(winners, dtype=np.int64), minlength=m)
    return int((wins > 0).sum()), wins


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def run_sweep(model_name, layer, device, batch, limit):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name).eval().to(device)
    torch.set_grad_enabled(False)

    blk = mdl.transformer.h[layer]
    mlp = blk.mlp
    buf = {}
    if hasattr(mlp, "act"):
        target, post = mlp.act, None
        hook_path = f"transformer.h[{layer}].mlp.act"
    else:
        target, post = mlp.c_fc, torch.nn.functional.gelu
        hook_path = f"transformer.h[{layer}].mlp.c_fc (+gelu)"
    target.register_forward_hook(lambda mod, i, o: buf.__setitem__("A", o.detach()))

    def acts(ids_2d):
        mdl(input_ids=ids_2d)
        A = buf["A"]
        if post is not None:
            A = post(A)
        return A.float()

    m = mdl.config.n_inner or 4 * mdl.config.n_embd
    V = mdl.config.vocab_size if limit is None else min(limit, mdl.config.vocab_size)

    # --- BOS probe: the constant that sits under every Implicit input ---
    bos_act = acts(torch.tensor([[BOS]], device=device))[0, 0].cpu().numpy()

    neg = -np.inf
    exp_win = np.zeros(V, dtype=np.int32)
    imp_r_win = np.zeros(V, dtype=np.int32)
    imp_i_win = np.zeros(V, dtype=np.int32)
    exp_max = np.full(m, neg, dtype=np.float64)
    imp_max_all = np.full(m, neg, dtype=np.float64)
    imp_max_last = np.full(m, neg, dtype=np.float64)

    for lo in range(0, V, batch):
        hi = min(lo + batch, V)
        ids = torch.arange(lo, hi, device=device, dtype=torch.long)

        A = acts(ids.unsqueeze(1))                       # (B, 1, m)
        exp_win[lo:hi] = A[:, 0, :].argmax(1).cpu().numpy()
        exp_max = np.maximum(exp_max, A[:, 0, :].max(0).values.cpu().numpy())

        pair = torch.stack([torch.full_like(ids, BOS), ids], dim=1)
        A = acts(pair)                                   # (B, 2, m)
        imp_r_win[lo:hi] = A.max(dim=1).values.argmax(1).cpu().numpy()
        imp_i_win[lo:hi] = A[:, 1, :].argmax(1).cpu().numpy()
        imp_max_all = np.maximum(imp_max_all, A.reshape(-1, m).max(0).values.cpu().numpy())
        imp_max_last = np.maximum(imp_max_last, A[:, 1, :].max(0).values.cpu().numpy())

        if (lo // batch) % 20 == 0:
            print(f"  {hi}/{V}", file=sys.stderr, flush=True)

    toks = tok.convert_ids_to_tokens(list(range(V)))
    meta = dict(model=model_name, layer=layer, hook=hook_path, m=m, vocab=V,
                device=device)
    return dict(meta=meta, tokens=toks, bos_act=bos_act, m=m, V=V,
                exp_win=exp_win, imp_r_win=imp_r_win, imp_i_win=imp_i_win,
                exp_max=exp_max, imp_max_all=imp_max_all, imp_max_last=imp_max_last)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report(r):
    m, V = r["m"], r["V"]
    bos = r["bos_act"]
    bos_top = np.argsort(bos)[::-1][:5]

    print()
    print("=== BOS probe ===")
    print("Activation at position 1 of EVERY Implicit input. Constant.")
    print(f"  loudest: neuron {int(bos_top[0])} at {bos[bos_top[0]]:.4f}")
    print("  top 5  : " + ", ".join(f"{int(j)}={bos[j]:.3f}" for j in bos_top))

    shadowed = int((r["imp_r_win"] == int(bos_top[0])).sum())
    print(f"  single tokens landing on neuron {int(bos_top[0])} under imp_r: "
          f"{shadowed}/{V} ({shadowed/V:.1%})")
    print("  (if that fraction is large, this neuron is a structural default,")
    print("   not a semantic destination)")

    print()
    print("=== one-token reachability (COMPLETE for |s| = 1) ===")
    rows = [("exp_r == exp_i", r["exp_win"]),
            ("imp_r", r["imp_r_win"]),
            ("imp_i", r["imp_i_win"])]
    for name, w in rows:
        n, wins = summarise(w, m)
        top = np.argsort(wins)[::-1][:3]
        print(f"  {name:14} {n:5d}/{m}  ({n/m:5.1%})   "
              f"busiest: " + ", ".join(f"{int(j)}({int(wins[j])})" for j in top))
    print("  note: a single EXPLICIT token has one position, so Resonance and")
    print("        Inference coincide there by construction.")


def ghost_test(r, db_path):
    import sqlite3
    m = r["m"]
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    quads = {"imp_r": r["imp_max_all"], "imp_i": r["imp_max_last"],
             "exp_r": r["exp_max"], "exp_i": r["exp_max"]}

    print()
    print("=== ghost test: are never-reached neurons structurally quieter? ===")
    print("measured max activation over all 50,257 single-token inputs,")
    print("split by whether the corpus ever reached that neuron")
    print()
    print(f"{'quad':6} {'n_unreached':>11} {'med(reached)':>13} {'med(unreached)':>15} "
          f"{'z':>8} {'p':>10}")
    for q, maxact in quads.items():
        hits = np.zeros(m, dtype=np.int64)
        for j, h in db.execute(
                "select neuron_id, corpus_hits from MasterHitCounts where quadrant_key=?", (q,)):
            if 0 <= j < m:
                hits[j] = h
        reached = maxact[hits > 0]
        unreached = maxact[hits == 0]
        if len(unreached) == 0:
            print(f"{q:6} {0:>11}   (corpus reached every neuron)")
            continue
        _, z, p = rank_sum(reached, unreached)
        print(f"{q:6} {len(unreached):>11} {np.median(reached):13.4f} "
              f"{np.median(unreached):15.4f} {z:8.2f} {p:10.2e}")
    print()
    print("z is computed as rank_sum(reached, unreached):")
    print("  POSITIVE z + small p => reached neurons are louder,")
    print("                          i.e. unreached neurons are measurably quieter")
    print("  => evidence for structural shadowing rather than bad luck (not proof)")
    print("caution: exp_* rows are unreliable here. A single EXPLICIT token sits at")
    print("  position 0 with no context, which is a degenerate regime, so exp_max is")
    print("  a poor proxy for what a neuron can do in real multi-token explicit input.")


def write_tsv(r, prefix):
    meta = "# " + repr(r["meta"])
    with open(f"{prefix}_tokens.tsv", "w", encoding="utf-8") as fh:
        print(meta, file=fh)
        print("token_id\ttoken\texp\timp_r\timp_i", file=fh)
        for i in range(r["V"]):
            print(f"{i}\t{esc(r['tokens'][i])}\t{r['exp_win'][i]}\t"
                  f"{r['imp_r_win'][i]}\t{r['imp_i_win'][i]}", file=fh)

    _, w_e = summarise(r["exp_win"], r["m"])
    _, w_r = summarise(r["imp_r_win"], r["m"])
    _, w_i = summarise(r["imp_i_win"], r["m"])
    with open(f"{prefix}_neurons.tsv", "w", encoding="utf-8") as fh:
        print(meta, file=fh)
        print("neuron\tbos_act\texp_wins\timp_r_wins\timp_i_wins\t"
              "exp_max\timp_max_all\timp_max_last", file=fh)
        for j in range(r["m"]):
            print(f"{j}\t{r['bos_act'][j]:.6f}\t{int(w_e[j])}\t{int(w_r[j])}\t"
                  f"{int(w_i[j])}\t{r['exp_max'][j]:.6f}\t"
                  f"{r['imp_max_all'][j]:.6f}\t{r['imp_max_last'][j]:.6f}", file=fh)
    print(f"\nwrote {prefix}_tokens.tsv and {prefix}_neurons.tsv")


# --------------------------------------------------------------------------


def selftest():
    a = np.array([5.0, 6, 7, 8, 9])
    b = np.array([1.0, 2, 3, 4])
    U, z, p = rank_sum(a, b)
    assert U == 20.0, U                      # a dominates b completely
    assert z > 0 and p < 0.05, (z, p)

    U2, _, _ = rank_sum(b, a)
    assert U2 == 0.0, U2

    _, z3, p3 = rank_sum(np.ones(50), np.ones(50))
    assert math.isnan(z3) or abs(z3) < 1e-9, z3

    x = np.array([1.0, 1, 2, 2, 3])
    U4, _, _ = rank_sum(x, x)                # ties handled, symmetric
    assert abs(U4 - len(x) ** 2 / 2) < 1e-9, U4

    assert esc("a\tb\nc") == "a\\tb\\nc"
    assert esc("x\\y") == "x\\\\y"

    n, wins = summarise([0, 0, 3, 3, 3, 7], 10)
    assert n == 3 and wins[3] == 3 and wins[0] == 2, (n, wins)

    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="only first N tokens")
    ap.add_argument("--prefix", default="sweep")
    ap.add_argument("--db", default=None, help="znou_exchange.db, enables ghost test")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"sweeping layer {args.layer} on {device} ...", file=sys.stderr)
    r = run_sweep(args.model, args.layer, device, args.batch, args.limit)
    print("# " + repr(r["meta"]))
    report(r)
    write_tsv(r, args.prefix)
    if args.db:
        ghost_test(r, args.db)


if __name__ == "__main__":
    main()

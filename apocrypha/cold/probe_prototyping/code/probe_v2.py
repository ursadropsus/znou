"""
probe_v2.py — per-position trace of the argmax competition, per SPEC.md §1/§7.

probe.py reported the winner and one margin. This reports the whole race:
for each position, the running max of each watched neuron, who currently
leads, and by how much. A destination that changes hands mid-string is
visible as the row where the leader column changes.

Quadrant imp_r (bos=True, rho=R) throughout. §1.1 guarantees the running
max column is non-decreasing; a fall in that column is a bug.

  trace(s)                  auto-watches the winner and runner-up
  trace(s, watch=[20,1888]) watches whatever you name
  prefix_sweep(tail, [...]) one-token prefixes against a fixed tail
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ELL, BOS, N_CTX = 5, 50256, 1024
REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

tok = GPT2TokenizerFast.from_pretrained("gpt2", revision=REV, add_prefix_space=False)
mdl = GPT2LMHeadModel.from_pretrained("gpt2", revision=REV).to(torch.float32).eval()

_buf = {}
mdl.transformer.h[ELL].mlp.act.register_forward_hook(
    lambda m, i, o: _buf.__setitem__("A", o.detach())
)

assert int(torch.tensor([0.0, 1.0, 1.0]).argmax()) == 1, "backend violates tie rule"


def activations(s, bos=True):
    ids = tok(s)["input_ids"]
    if bos:
        ids = [BOS] + ids
    assert 0 < len(ids) <= N_CTX, "outside dom(D)"
    with torch.inference_mode():
        mdl(input_ids=torch.tensor([ids]))
    A = _buf["A"][0]
    assert A.dtype is torch.float32
    return ids, A


def destination(s, bos=True):
    """(j, delta) under rho=R."""
    _, A = activations(s, bos)
    v = A.max(0).values
    top2 = torch.topk(v, 2)
    return int(top2.indices[0]), float(top2.values[0] - top2.values[1])


def trace(s, watch=None, k=3, bos=True):
    """Per-position running-max trace of the competition."""
    ids, A = activations(s, bos)
    T = A.shape[0]
    final = A.max(0).values
    top2 = torch.topk(final, 2)
    winner, runner = int(top2.indices[0]), int(top2.indices[1])

    if watch is None:
        watch = [winner, runner]
    watch = list(dict.fromkeys(watch))          # dedupe, keep order

    print(f"{s!r}")
    print(f"  final: j={winner}  delta={float(top2.values[0]-top2.values[1]):.4f}"
          f"   runner-up {runner}")

    head = f"  {'pos':>3} {'token':<12}" + "".join(f"{'j='+str(j):>10}" for j in watch)
    head += f"{'leader':>8}{'delta':>9}   top-{k} at this position"
    print(head)
    print("  " + "-" * (len(head) - 2))

    prev_leader = None
    for t in range(T):
        run = A[: t + 1, :].max(0).values       # running max over positions <= t
        lead2 = torch.topk(run, 2)
        leader = int(lead2.indices[0])
        d = float(lead2.values[0] - lead2.values[1])

        here = torch.topk(A[t, :], k)
        here_s = " ".join(f"{int(i)}:{float(v):+.3f}"
                          for i, v in zip(here.indices, here.values))

        label = tok.decode([ids[t]])
        if t == 0 and bos:
            label = "<BOS>"

        cols = "".join(f"{float(run[j]):>10.4f}" for j in watch)
        mark = " *" if leader != prev_leader else "  "
        print(f"  {t:>3} {label!r:<12}{cols}{leader:>8}{d:>9.4f}{mark} {here_s}")
        prev_leader = leader

    print("  (* = lead changed hands at this position)\n")


def prefix_sweep(tail, prefixes, watch_j=None, bos=True):
    """
    Each prefix + fixed tail, plus the prefix alone, plus that prefix
    token's position-1 activation on watch_j. Tests whether position-1
    loudness on the incumbent predicts whether the tail's neuron gets through.
    """
    tail_j, tail_d = destination(tail, bos)
    print(f"tail {tail!r} alone -> {tail_j}  delta {tail_d:.4f}")
    if watch_j is None:
        watch_j = tail_j
    print(f"  {'prefix':<14}{'joined':>8}{'delta':>9}{'alone':>8}"
          f"{'delta':>9}{'A[1,'+str(watch_j)+']':>12}")
    print("  " + "-" * 60)

    rows = []
    for p in prefixes:
        joined = p + tail
        jj, jd = destination(joined, bos)
        pj, pd = destination(p, bos)
        _, Ap = activations(p, bos)
        a1 = float(Ap[1, watch_j]) if Ap.shape[0] > 1 else float("nan")
        rows.append((p, jj, jd, pj, pd, a1))
        print(f"  {p!r:<14}{jj:>8}{jd:>9.4f}{pj:>8}{pd:>9.4f}{a1:>12.4f}")

    hit = [r for r in rows if r[1] == tail_j]
    print(f"\n  {len(hit)}/{len(rows)} prefixes preserve {tail_j}")
    if hit and len(hit) < len(rows):
        miss = [r for r in rows if r[1] != tail_j]
        print(f"  A[1,{watch_j}] among preservers: "
              f"{min(r[5] for r in hit):.3f} .. {max(r[5] for r in hit):.3f}")
        print(f"  A[1,{watch_j}] among failures:   "
              f"{min(r[5] for r in miss):.3f} .. {max(r[5] for r in miss):.3f}")
    print()


if __name__ == "__main__":
    print("=== traces ===\n")
    trace(" Mackinaw", watch=[20, 1888])
    trace("at Mackinaw", watch=[20, 1888])
    trace("in Mackinaw", watch=[20, 1888])
    trace("Upon my word were I at Mackinaw", watch=[20, 1888])

    print("=== prefix sweep ===\n")
    prefix_sweep(
        " Mackinaw",
        ["at", "the", "in", "I at", "on", "near", "a", "was", "from",
         "xq", "The", "to", "of"],
        watch_j=1888,
    )

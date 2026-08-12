"""
replay_cache.py  —  J-Space §7.1 arm 1 / TODO A2

Replays a precomputed atlas cache through the CURRENT stack and diffs
per sentence. The cache was generated on a different machine at a different
time, so this is a real cross-environment measurement, not a regression test.

Reports agreement at BOTH levels, because they are not the same claim:
  - per sentence   (7,353 comparisons)  <- the real test
  - per system set (545 distinct)       <- what §8 currently reports

Usage:
    python replay_cache.py path/to/the_sea_implicit_resonance.json
    python replay_cache.py path/to/cache.json --bos 0 --rho I
"""

import argparse, hashlib, json, platform, sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ELL, BOS, N_CTX = 5, 50256, 1024

ap = argparse.ArgumentParser()
ap.add_argument("cache")
ap.add_argument("--bos", type=int, default=1, choices=[0, 1])
ap.add_argument("--rho", default="R", choices=["R", "I"])
ap.add_argument("--out", default="replay_results.tsv")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

# ---- pin the numerics exactly as §7 requires -------------------------------
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

tok = GPT2TokenizerFast.from_pretrained("gpt2")
mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.float32).eval().to(args.device)

buf = {}
mdl.transformer.h[ELL].mlp.act.register_forward_hook(
    lambda m, i, o: buf.__setitem__("A", o.detach())
)


def destination(s):
    """Returns (system, margin). Margin = A_max - A_2nd over J."""
    ids = tok(s)["input_ids"]
    if args.bos:
        ids = [BOS] + ids
    if not (0 < len(ids) <= N_CTX):
        return None, None                       # outside dom(D), §1 MUST
    with torch.inference_mode():
        mdl(input_ids=torch.tensor([ids], device=args.device))
    A = buf["A"][0]
    assert A.dtype is torch.float32, "forward was not fp32"
    v = A.max(0).values if args.rho == "R" else A[-1]
    top2 = torch.topk(v, 2)
    return int(top2.indices[0]), float(top2.values[0] - top2.values[1])


# ---- environment record ----------------------------------------------------
env = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": args.device,
    "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "n/a",
    "platform": platform.platform(),
    "cache_sha256": hashlib.sha256(open(args.cache, "rb").read()).hexdigest()[:16],
}
print("\n".join(f"{k:14} {v}" for k, v in env.items()), "\n")

rows = json.load(open(args.cache, encoding="utf-8"))
print(f"replaying {len(rows)} entries from {args.cache}\n")

match, mismatch, skipped = [], [], 0
with open(args.out, "w", encoding="utf-8") as f:
    f.write("idx\tstored\tcurrent\tdelta\tagree\tsentence\n")
    for i, r in enumerate(rows):
        sent, stored = r["sentence"], r["neuron_id"]
        cur, delta = destination(sent)
        if cur is None:
            skipped += 1
            continue
        agree = cur == stored
        (match if agree else mismatch).append(delta)
        f.write(f"{i}\t{stored}\t{cur}\t{delta:.6f}\t{int(agree)}\t{sent[:120]}\n")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)}  mismatches so far: {len(mismatch)}")

n = len(match) + len(mismatch)


def pct(x, d):
    return f"{100*x/d:.3f}%" if d else "n/a"


print(f"\n{'='*58}\nPER-SENTENCE  (the real test)")
print(f"  compared      {n}")
print(f"  agree         {len(match)}   ({pct(len(match), n)})")
print(f"  DISAGREE      {len(mismatch)}   ({pct(len(mismatch), n)})")
if skipped:
    print(f"  skipped       {skipped}  (outside dom(D))")

stored_set = {r["neuron_id"] for r in rows}
current_set = set()
for line in open(args.out, encoding="utf-8").readlines()[1:]:
    current_set.add(int(line.split("\t")[2]))

print(f"\nPER-SYSTEM SET  (what §8 currently reports)")
print(f"  stored distinct   {len(stored_set)}")
print(f"  current distinct  {len(current_set)}")
print(f"  set difference    +{len(current_set - stored_set)} / -{len(stored_set - current_set)}")

if mismatch:
    mismatch.sort()
    match.sort()
    q = lambda a, p: a[min(int(p * len(a)), len(a) - 1)]
    print(f"\nMARGIN OF DISAGREEING SENTENCES (current-stack Δ)")
    print(f"  min {mismatch[0]:.4f} · med {q(mismatch,.5):.4f} · max {mismatch[-1]:.4f}")
    print(f"  agreeing:  med {q(match,.5):.4f}")
    print(f"\n  Δ<0.01 : {sum(d<0.01 for d in mismatch)}/{len(mismatch)} of disagreements")
    print(f"           {sum(d<0.01 for d in match)}/{len(match)} of agreements")
    print("\n  If disagreements concentrate at low Δ, I5 holds and §6.1 has a")
    print("  mechanic. If they are spread across Δ, something other than float")
    print("  noise differs between the two runs — check the generating script")
    print("  implements the same D before concluding anything.")
else:
    print(f"\n  Zero disagreements across {n} sentences on a different machine")
    print("  and a different stack. Cross-environment perturbation is below the")
    print("  smallest margin present in this corpus. §6.1's mechanic does not")
    print("  exist under fp32 and should be withdrawn rather than softened.")

print(f"\nwrote {args.out}\n{'='*58}")

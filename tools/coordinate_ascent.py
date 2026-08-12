"""
coordinate_ascent.py — attack the unreached set of J.

Gradient-guided discrete coordinate ascent (GCG-style) over token sequences,
targeting a neuron j at layer ELL under a given (beta_bos, rho) quadrant.

Reports, per target:

    m_tok    margin at the token level (search space is V*)
    m_str    margin of D on the DECODED STRING (search space is Sigma*)

Only m_str > 0 is a certificate for j in R. dom(D) is Sigma*, not V*: the
optimiser works over token sequences, but tau is not surjective onto V*, so a
token-level hit must be re-evaluated on the string it decodes to. Note the
re-encoding need not equal the ids found -- it only has to land on j.

By default the candidate pool is restricted to round-trip-safe tokens
(valid UTF-8, re-encoding to themselves), which excludes byte-fragment
tokens. Pass --no-safe to search the full vocabulary.

Usage
-----
  # smoke test
  python coordinate_ascent.py --targets rand:20 --quadrant imp_r

  # POSITIVE CONTROL -- clearance rate on neurons known to be in R-hat_C
  python coordinate_ascent.py --targets reached_imp_r.txt --sample 100 \
      --quadrant imp_r --out control_imp_r.tsv

  # the real run
  python coordinate_ascent.py --targets unreached_imp_r.txt \
      --quadrant imp_r --len 16 --sweeps 8 --restarts 4 --out ascent_imp_r.tsv

  # margin readout for existing fixtures (one string per line)
  python coordinate_ascent.py --fixtures typed_lines.txt --quadrant imp_r
"""

import argparse, os, random, sys, time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ELL, BOS, M, N_CTX = 5, 50256, 3072, 1024
SAFE_CACHE = "safe_tokens.pt"


# --------------------------------------------------------------------------
def build(device, dtype):
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    full = GPT2LMHeadModel.from_pretrained("gpt2")
    mdl = full.transformer
    mdl.h = torch.nn.ModuleList(list(mdl.h[: ELL + 1]))   # blocks 0..ELL only
    mdl = mdl.to(device=device, dtype=dtype).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    buf = {}
    mdl.h[ELL].mlp.act.register_forward_hook(
        lambda m, i, o: buf.__setitem__("A", o)
    )
    return tok, mdl, buf


def score(A, rho):
    """A: (B, T, M) -> (B, M) readout."""
    return A.max(dim=1).values if rho == "R" else A[:, -1, :]


def margin(S, j):
    """S: (B, M). Signed margin of j over its best competitor. >0 == reached."""
    tgt = S[:, j].clone()
    other = S.clone()
    other[:, j] = float("-inf")
    return tgt - other.max(dim=1).values


def safe_token_mask(tok, V):
    """Tokens decoding to valid UTF-8 that re-encode to themselves."""
    if os.path.exists(SAFE_CACHE):
        return torch.load(SAFE_CACHE)
    keep = torch.zeros(V, dtype=torch.bool)
    for v in range(V):
        s = tok.decode([v])
        if "\ufffd" in s or not s:
            continue
        if tok(s)["input_ids"] == [v]:
            keep[v] = True
    torch.save(keep, SAFE_CACHE)
    return keep


@torch.no_grad()
def eval_string(s, j, tok, mdl, buf, *, rho, bos, device):
    """Margin of j on the actual string s. This is the certificate test."""
    ids = tok(s)["input_ids"]
    if bos:
        ids = [BOS] + ids
    if not ids or len(ids) > N_CTX:
        return None
    mdl(input_ids=torch.tensor([ids], device=device))
    return margin(score(buf["A"].float(), rho), j)[0].item()


# --------------------------------------------------------------------------
def ascend(j, mdl, buf, *, rho, bos, n_tok, k, sweeps, device, dtype, rng,
           mask, init=None):
    """One restart. Returns (best_margin, best_ids)."""
    V = mdl.wte.weight.shape[0]
    pool = mask.nonzero(as_tuple=True)[0] if mask is not None else None
    draw = (lambda: int(pool[rng.randrange(len(pool))])) if pool is not None \
        else (lambda: rng.randrange(V))

    if init is not None:
        best = init.clone().to(device)
        n_tok = int(best.numel())
    else:
        best = torch.tensor([draw() for _ in range(n_tok)], device=device)
    prefix = torch.tensor([BOS], device=device) if bos else None

    def full_ids(x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if prefix is None:
            return x
        return torch.cat([prefix.expand(x.shape[0], 1), x], dim=1)

    @torch.no_grad()
    def evaluate(cand):                      # cand: (B, n_tok) -> (B,)
        out = []
        for s in range(0, cand.shape[0], 256):
            mdl(input_ids=full_ids(cand[s : s + 256]))
            out.append(margin(score(buf["A"].float(), rho), j))
        return torch.cat(out)

    best_m = evaluate(best.unsqueeze(0))[0].item()

    for _ in range(sweeps):
        for p in rng.sample(range(n_tok), n_tok):
            oh = torch.zeros(n_tok, V, device=device, dtype=dtype)
            oh[torch.arange(n_tok), best] = 1.0
            oh.requires_grad_(True)
            emb = oh @ mdl.wte.weight
            if prefix is not None:
                emb = torch.cat([mdl.wte(prefix), emb], dim=0)
            mdl(inputs_embeds=emb.unsqueeze(0))
            (-margin(score(buf["A"].float(), rho), j)[0]).backward()

            sc = -oh.grad[p]                                 # steepest ascent
            if mask is not None:
                sc = sc.masked_fill(~mask, float("-inf"))
            cand = best.unsqueeze(0).repeat(k, 1)
            cand[:, p] = torch.topk(sc, k).indices
            m = evaluate(cand)
            i = int(m.argmax())
            if m[i].item() > best_m:
                best_m, best = m[i].item(), cand[i].clone()
            if best_m > 0:
                return best_m, best
    return best_m, best


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets",
                    help="file of neuron indices (one per line), 'all', or 'rand:N'")
    ap.add_argument("--fixtures", help="file of strings, one per line")
    ap.add_argument("--sample", type=int, default=0, help="0 = all targets")
    ap.add_argument("--quadrant", default="imp_r",
                    choices=["imp_r", "imp_i", "exp_r", "exp_i"])
    ap.add_argument("--len", type=int, default=8, dest="n_tok")
    ap.add_argument("--k", type=int, default=192, help="candidates per position")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--no-safe", action="store_true",
                    help="search the full vocabulary incl. byte fragments")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    a = ap.parse_args()

    if not (a.targets or a.fixtures):
        ap.error("need --targets (file | 'all' | 'rand:N') or --fixtures FILE")

    bos = a.quadrant.startswith("imp")
    rho = "R" if a.quadrant.endswith("_r") else "I"
    dtype = torch.float32                       # fp32 forward; see spec I5
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    tok, mdl, buf = build(a.device, dtype)

    # ---- fixture margin readout ------------------------------------------
    if a.fixtures:
        print("dest\tmargin\tstring")
        with torch.no_grad():
            for line in open(a.fixtures, encoding="utf-8"):
                s = line.rstrip("\n")
                ids = tok(s)["input_ids"]
                if bos:
                    ids = [BOS] + ids
                mdl(input_ids=torch.tensor([ids], device=a.device))
                S = score(buf["A"].float(), rho)[0]
                top = S.topk(2).values
                print(f"{int(S.argmax())}\t{(top[0]-top[1]).item():.4f}\t{s}")
        return

    mask = None
    if not a.no_safe:
        t = time.time()
        mask = safe_token_mask(tok, mdl.wte.weight.shape[0]).to(a.device)
        print(f"# safe tokens {int(mask.sum())}/{mask.numel()} "
              f"({time.time()-t:.0f}s)", file=sys.stderr)

    if a.targets == "all":
        targets = list(range(M))
    elif a.targets.startswith("rand:"):
        targets = rng.sample(range(M), int(a.targets.split(":", 1)[1]))
    else:
        targets = [int(x) for x in open(a.targets) if x.strip()]
    if a.sample:
        targets = rng.sample(targets, min(a.sample, len(targets)))

    # resume: skip targets already present in --out, and append
    fresh = True
    if a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        done = {int(p[0]) for p in
                (l.split('\t') for l in open(a.out, encoding='utf-8'))
                if p and p[0].isdigit()}
        if done:
            targets = [j for j in targets if j not in done]
            fresh = False
            print(f"# resuming: {len(done)} done, {len(targets)} left",
                  file=sys.stderr)

    out = open(a.out, "a" if not fresh else "w", encoding="utf-8") \
        if a.out else sys.stdout
    if fresh:
        print("j\thit_tok\thit_str\tm_tok\tm_str\tstring", file=out)

    n_tok_hit = n_str_hit = 0
    t0 = time.time()
    for n, j in enumerate(targets, 1):
        best_m, best_ids = -1e9, None
        for _ in range(a.restarts):
            m, ids = ascend(j, mdl, buf, rho=rho, bos=bos, n_tok=a.n_tok,
                            k=a.k, sweeps=a.sweeps, device=a.device,
                            dtype=dtype, rng=rng, mask=mask)
            if m > best_m:
                best_m, best_ids = m, ids
            if best_m > 0:
                break

        s = tok.decode(best_ids.tolist())
        m_str = eval_string(s, j, tok, mdl, buf,
                            rho=rho, bos=bos, device=a.device)

        # repair: token-level hit that isn't a string yet. Re-encode and
        # restart the search from there -- it is close to the manifold.
        if best_m > 0 and (m_str is None or m_str <= 0):
            ids2 = tok(s)["input_ids"]
            if ids2 and len(ids2) + int(bos) <= N_CTX:
                m2, i2 = ascend(j, mdl, buf, rho=rho, bos=bos, n_tok=a.n_tok,
                                k=a.k, sweeps=a.sweeps, device=a.device,
                                dtype=dtype, rng=rng, mask=mask,
                                init=torch.tensor(ids2, device=a.device))
                s2 = tok.decode(i2.tolist())
                m2s = eval_string(s2, j, tok, mdl, buf,
                                  rho=rho, bos=bos, device=a.device)
                if m2s is not None and (m_str is None or m2s > m_str):
                    s, m_str, best_m = s2, m2s, max(best_m, m2)

        hit_t, hit_s = best_m > 0, (m_str is not None and m_str > 0)
        n_tok_hit += hit_t
        n_str_hit += hit_s
        ms = "NA" if m_str is None else f"{m_str:.4f}"
        print(f"{j}\t{int(hit_t)}\t{int(hit_s)}\t{best_m:.4f}\t{ms}\t{s!r}",
              file=out, flush=True)
        if n % 10 == 0:
            print(f"  [{n}/{len(targets)}] tok {n_tok_hit} str {n_str_hit} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)

    print(f"\n# targets {len(targets)}  hit_tok {n_tok_hit}  "
          f"hit_str {n_str_hit}  ({time.time()-t0:.0f}s)", file=out)


if __name__ == "__main__":
    main()
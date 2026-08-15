"""
probe_v3.py — is 20 gated on the 'in' continuation token, or on ' Mack'+'in'?

Three blocks, all imp_r (bos=True, rho=R), all with a fixed carrier prefix
so nothing sits at position 1:

  A. many stems, same 'in' continuation   — does any stem + 'in' reach 20?
  B. one stem, many continuations         — is ' Mack' doing the work?
  C. same word, many carriers             — how much does the prefix move it?

Column A[t,20] is the RAW per-position activation at the 'in' token, not a
running max. That is the quantity that has to clear the incumbent.

Words that tokenize without a separate 'in' token are not failures; they are
controls, and are reported with 'in' position as '-'.
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

IN_IDS = tok("in")["input_ids"]
assert len(IN_IDS) == 1
IN_ID = IN_IDS[0]


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


def row(s, watch, target_id=None, bos=True):
    """One line: tokens, position of target token, raw activations there, dest."""
    ids, A = activations(s, bos)
    v = A.max(0).values
    top2 = torch.topk(v, 2)
    j, delta = int(top2.indices[0]), float(top2.values[0] - top2.values[1])

    body = ids[1:] if bos else ids
    toks = [tok.decode([i]) for i in body]

    # positions (in full-sequence indexing) where the target token occurs
    hits = [k for k, i in enumerate(ids) if i == target_id and not (bos and k == 0)]
    t = hits[-1] if hits else None       # last occurrence: the in-word one

    if t is None:
        acts = ["-"] * len(watch)
        tpos = "-"
    else:
        acts = [f"{float(A[t, w]):+.4f}" for w in watch]
        tpos = str(t)

    return toks, tpos, acts, j, delta


def table(title, strings, watch, target_id=IN_ID, note=""):
    print(f"--- {title} ---")
    if note:
        print(f"    {note}")
    head = (f"  {'string':<20}{'pos':>5}"
            + "".join(f"{'A[t,'+str(w)+']':>12}" for w in watch)
            + f"{'dest':>7}{'delta':>9}  tokens")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for s in strings:
        toks, tpos, acts, j, d = row(s, watch, target_id)
        acol = "".join(f"{a:>12}" for a in acts)
        print(f"  {s!r:<20}{tpos:>5}{acol}{j:>7}{d:>9.4f}  {toks}")
    print()


WATCH = [20, 1430, 1888]
CARRIER = "at "

# A. many stems, same continuation. Nonsense stems included on purpose:
#    if 20 needs a real place name, they should fail.
STEMS = ["Mackinaw", "Mackinac", "Franklin", "Dublin", "Berlin", "Austin",
         "Merlin", "cabin", "napkin", "pumpkin", "margin", "basin",
         "robin", "goblin", "muffin", "coffin", "Latin",
         "Zorkinaw", "Blarkinaw", "Quibinaw", "Trupin", "Fenrisin"]

# B. one stem, many continuations. Isolates ' Mack' from 'in'.
MACKS = ["Mackinaw", "Mackinac", "Mackintosh", "Macklin", "Mackay",
         "Mackerel", "Mackle", "Mackson", "Mack"]

# C. same word, many carriers. Spread here bounds how much the prefix matters.
CARRIERS = ["at ", "the ", "on ", "near ", "was ", "from ", "of ", "to ",
            "I at ", "The ", "xq ", "in "]


if __name__ == "__main__":
    print(f"'in' continuation token id = {IN_ID}\n")

    table("A. stems + 'in', carrier 'at '",
          [CARRIER + w for w in STEMS], WATCH,
          note="if nonsense stems reach 20, the stem is not what matters")

    table("B. ' Mack' + various continuations, carrier 'at '",
          [CARRIER + w for w in MACKS], WATCH,
          note="'-' in pos means no separate 'in' token in this word")

    table("C. 'Mackinaw' under various carriers",
          [c + "Mackinaw" for c in CARRIERS], WATCH,
          note="A[t,20] spread here is the context sensitivity of 20 itself")

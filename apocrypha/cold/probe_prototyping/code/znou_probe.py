"""
znou_probe.py — shared stack, measurement and recording for the probe series.

Everything here is imp_r-capable but quadrant-agnostic; pass bos/rho.
Pins follow SPEC.md §7. Import this from probe_v4 onward; probe.py through
probe_v3.py are self-contained on purpose and are the iteration trail.

  activations(s)            (ids, A)  with A the (T, 3072) post-GELU tensor
  measure(s, watch=[...])   one dict per string, wide format
  trace_rows(s, watch=...)  one dict per position, long format
  Recorder(name, dir=...)   accumulates dicts, writes a self-describing TSV
"""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ------------------------------------------------------------------- §7 pins

ELL, BOS, N_CTX = 5, 50256, 1024
REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

tok = GPT2TokenizerFast.from_pretrained("gpt2", revision=REV, add_prefix_space=False)
mdl = GPT2LMHeadModel.from_pretrained("gpt2", revision=REV)
mdl = mdl.to(device=DEVICE, dtype=torch.float32).eval()

_buf = {}
mdl.transformer.h[ELL].mlp.act.register_forward_hook(
    lambda m, i, o: _buf.__setitem__("A", o.detach())
)

# §1 tie rule: lowest index wins. Asserted, not inherited from the backend.
assert int(torch.tensor([0.0, 1.0, 1.0]).argmax()) == 1, "backend violates tie rule"
assert int(torch.tensor([0.0, 1.0, 1.0], device=DEVICE).argmax()) == 1, \
    "compute device violates tie rule"

_theta_sha = None


def theta_sha256():
    """
    sha256 over the state dict — names, dtypes, shapes, bytes, sorted.
    §7 pins this value; tools/pin_stack.py is the reference implementation
    and this is byte-for-byte the same procedure.

    No per-tensor cast: the model is already fp32 from the .to() at load,
    and casting here would also convert any bool/int buffer, changing both
    its dtype string and its bytes relative to pin_stack.py.

    Note that pin_stack.py loads WITHOUT revision= while this module loads
    WITH it. If the two digests differ, the local HF cache is not resolving
    'gpt2' to the pinned revision and §7's published value needs regenerating
    rather than this one being wrong.
    """
    global _theta_sha
    if _theta_sha is None:
        h = hashlib.sha256()
        sd = mdl.state_dict()
        for k in sorted(sd):
            t = sd[k].detach().cpu().contiguous()
            h.update(k.encode())
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
        _theta_sha = h.hexdigest()
    return _theta_sha


def stack_metadata(theta_hash=True):
    actual_device = next(mdl.parameters()).device
    md = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(sys.argv[0]).name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(actual_device),
        "gpu": (torch.cuda.get_device_name(actual_device)
                if actual_device.type == "cuda" else "-"),
        "revision": REV,
        "layer": ELL,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if theta_hash:
        md["theta_sha256"] = theta_sha256()
    return md


# -------------------------------------------------------------- measurement

def activations(s, bos=True):
    ids = tok(s)["input_ids"]
    if bos:
        ids = [BOS] + ids
    assert 0 < len(ids) <= N_CTX, "outside dom(D)"      # §1 MUST
    with torch.inference_mode():
        mdl(input_ids=torch.tensor([ids], device=DEVICE))
    A = _buf["A"][0]
    assert A.dtype is torch.float32
    return ids, A


def _readout(A, rho="R"):
    return A.max(0).values if rho == "R" else A[-1]


def destination(s, bos=True, rho="R"):
    _, A = activations(s, bos)
    v = _readout(A, rho)
    top2 = torch.topk(v, 2)
    return int(top2.indices[0]), float(top2.values[0] - top2.values[1])


def measure(s, watch=(), bos=True, rho="R", with_prefix=True, **extra):
    """
    One wide row per string. `watch` adds peak_<j> and peakpos_<j> columns.
    with_prefix adds the §1.1 truncation at t*, decoded and re-evaluated.
    """
    ids, A = activations(s, bos)
    v = _readout(A, rho)
    top2 = torch.topk(v, 2)
    j = int(top2.indices[0])
    delta = float(top2.values[0] - top2.values[1])
    runner = int(top2.indices[1])
    t_star = int(A[:, j].argmax())

    body = ids[1:] if bos else ids
    toks = [tok.decode([i]) for i in body]

    row = {
        "string": s,
        "bos": int(bos),
        "rho": rho,
        "n_tok": len(body),
        "dest": j,
        "delta": round(delta, 6),
        "t_star": t_star,
        "runner_up": runner,
        "peak_val": round(float(top2.values[0]), 6),
        "tokens": json.dumps(toks, ensure_ascii=False),
        "token_ids": json.dumps(body),
    }

    for w in watch:
        row[f"peak_{w}"] = round(float(A[:, w].max()), 6)
        row[f"peakpos_{w}"] = int(A[:, w].argmax())

    if with_prefix and rho == "R":
        if bos and t_star == 0:
            row.update(prefix="", prefix_dest="", prefix_delta="",
                       prefix_agrees="bos_footprint")
        else:
            cut = t_star if bos else t_star + 1
            pre = tok.decode(body[:cut])
            pj, pd = destination(pre, bos, rho) if pre else ("", "")
            row.update(prefix=pre, prefix_dest=pj,
                       prefix_delta=round(pd, 6) if pre else "",
                       prefix_agrees=int(pj == j) if pre else "")

    row.update(extra)
    return row


def trace_rows(s, watch=(), k=3, bos=True, **extra):
    """Long format: one row per position. Running max is non-decreasing (§1.1)."""
    ids, A = activations(s, bos)
    out = []
    for t in range(A.shape[0]):
        run = A[: t + 1, :].max(0).values
        lead2 = torch.topk(run, 2)
        here = torch.topk(A[t, :], k)
        r = {
            "string": s,
            "pos": t,
            "token": "<BOS>" if (bos and t == 0) else tok.decode([ids[t]]),
            "token_id": ids[t],
            "leader": int(lead2.indices[0]),
            "leader_delta": round(float(lead2.values[0] - lead2.values[1]), 6),
            "topk": json.dumps([[int(i), round(float(x), 6)]
                                for i, x in zip(here.indices, here.values)]),
        }
        for w in watch:
            r[f"raw_{w}"] = round(float(A[t, w]), 6)
            r[f"run_{w}"] = round(float(run[w]), 6)
        r.update(extra)
        out.append(r)
    return out


# ----------------------------------------------------------------- recording

class Recorder:
    """
    Accumulates dicts, writes one TSV under results/ with a commented header
    carrying the stack metadata. Never overwrites: filename is timestamped.
    Fields are the union of all row keys, ordered by first appearance.
    """

    def __init__(self, name, outdir=None, theta_hash=True):
        self.name = name
        self.rows = []
        self.theta_hash = theta_hash
        self.outdir = Path(outdir) if outdir else Path(__file__).resolve().parent.parent / "results"

    def add(self, row):
        self.rows.append(row)
        return row

    def extend(self, rows):
        self.rows.extend(rows)
        return rows

    def write(self, suffix=""):
        if not self.rows:
            print(f"  [{self.name}] nothing to write")
            return None
        self.outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"_{suffix}" if suffix else ""
        path = self.outdir / f"{self.name}{tag}_{stamp}.tsv"

        fields = []
        for r in self.rows:
            for k in r:
                if k not in fields:
                    fields.append(k)

        with open(path, "w", encoding="utf-8", newline="") as f:
            for k, v in stack_metadata(self.theta_hash).items():
                f.write(f"# {k}\t{v}\n")
            f.write(f"# n_rows\t{len(self.rows)}\n")
            f.write("\t".join(fields) + "\n")
            for r in self.rows:
                f.write("\t".join(_cell(r.get(k, "")) for k in fields) + "\n")

        print(f"  wrote {path}  ({len(self.rows)} rows)")
        return path


def _cell(v):
    s = str(v)
    return s.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "")

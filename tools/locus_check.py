#!/usr/bin/env python3
"""
locus_check.py -- resolve lines of text to GPT-2 MLP neuron indices.

Standalone. Does not touch znou_exchange.db or any game code. Read-only:
it loads GPT-2, runs a forward pass per line, reports where each line lands.

Usage:
    python locus_check.py script.txt
    python locus_check.py script.txt --mode all,last -o out.tsv
    python locus_check.py script.txt --md            # edit-friendly card
    python locus_check.py --selftest                 # no model needed

Input file: one input string per line. Blank lines are skipped.
By default the script REFUSES to run if any line has trailing whitespace,
because a trailing space is part of the key and will change the destination.
"""

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Pure logic (no torch) -- kept separate so it can be tested without the model.
# ---------------------------------------------------------------------------

MODES = ("all", "last")


def pick_neuron(acts, mode):
    """acts: 2-D sequence, shape (n_tokens, n_units). Returns unit index.

    'all'  -> unit with the single highest activation anywhere in the sentence
    'last' -> unit with the highest activation at the final token position
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if len(acts) == 0:
        raise ValueError("empty activation matrix")

    if mode == "last":
        row = acts[-1]
        return max(range(len(row)), key=lambda j: row[j])

    best_val = None
    best_idx = None
    for row in acts:
        for j, v in enumerate(row):
            if best_val is None or v > best_val:
                best_val, best_idx = v, j
    return best_idx


def find_trailing_ws(lines):
    """Return [(1-based line no, text)] for lines with trailing whitespace."""
    return [(i, t) for i, t in enumerate(lines, 1) if t != t.rstrip()]


def read_lines(path, strip):
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read().split("\n")
    out = []
    for t in raw:
        t = t.rstrip("\r")
        if strip:
            t = t.rstrip()
        if t.strip() == "":
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Resolver:
    def __init__(self, model_name="gpt2", layer=5, one_indexed=False, device="cpu",
                 prepend_bos=False, skip_first=False):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.prepend_bos = prepend_bos
        self.skip_first = skip_first

        self.tok = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        torch.set_grad_enabled(False)

        blocks = self.model.transformer.h
        idx = layer - 1 if one_indexed else layer
        if not (0 <= idx < len(blocks)):
            raise SystemExit(
                f"layer {layer} out of range: model has {len(blocks)} blocks "
                f"(0-{len(blocks)-1} zero-indexed)"
            )
        self.layer_idx = idx
        self.n_blocks = len(blocks)

        mlp = blocks[idx].mlp
        if hasattr(mlp, "act"):
            self.target = mlp.act
            self.hook_path = f"transformer.h[{idx}].mlp.act"
            self._post = None
        else:  # fall back to pre-activation + gelu
            self.target = mlp.c_fc
            self.hook_path = f"transformer.h[{idx}].mlp.c_fc (+gelu)"
            self._post = torch.nn.functional.gelu

        self.n_units = self.model.config.n_inner or 4 * self.model.config.n_embd
        self._buf = {}
        self.target.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self._buf["acts"] = out.detach()

    def resolve(self, text, modes):
        enc = self.tok(text)["input_ids"]
        n_tok = len(enc)
        if n_tok == 0:
            return {m: None for m in modes}, 0
        if self.prepend_bos:
            enc = [self.tok.bos_token_id] + enc
        if len(enc) > self.model.config.n_positions:
            raise SystemExit(f"input too long ({len(enc)} tokens): {text[:60]!r}")

        ids = self.torch.tensor([enc], device=self.device)
        self.model(input_ids=ids)
        acts = self._buf["acts"]
        if self._post is not None:
            acts = self._post(acts)
        acts = acts[0].float()  # (n_positions, n_units)

        # optionally drop position 0, which carries a large context-free artefact
        pool = acts[1:] if (self.skip_first and acts.shape[0] > 1) else acts

        res = {}
        if "all" in modes:
            flat = int(self.torch.argmax(pool).item())
            res["all"] = flat % pool.shape[-1]
        if "last" in modes:
            res["last"] = int(self.torch.argmax(acts[-1]).item())
        return res, n_tok

    def meta(self):
        import torch
        import transformers

        return {
            "model": self.model_name,
            "block_index_zero_based": self.layer_idx,
            "n_blocks": self.n_blocks,
            "hook": self.hook_path,
            "n_units": self.n_units,
            "prepend_bos": self.prepend_bos,
            "skip_first_position": self.skip_first,
            "device": self.device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest():
    acts = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 9.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
    ]
    assert pick_neuron(acts, "all") == 2, pick_neuron(acts, "all")
    assert pick_neuron(acts, "last") == 0, pick_neuron(acts, "last")

    single = [[5.0, 5.0, 1.0]]
    assert pick_neuron(single, "all") == 0
    assert pick_neuron(single, "last") == 0

    neg = [[-3.0, -1.0], [-7.0, -8.0]]
    assert pick_neuron(neg, "all") == 1
    assert pick_neuron(neg, "last") == 0

    ws = find_trailing_ws(["ok", "bad ", "also bad\t", "fine"])
    assert [n for n, _ in ws] == [2, 3], ws

    try:
        pick_neuron([], "all")
    except ValueError:
        pass
    else:
        raise AssertionError("empty input should raise")

    try:
        pick_neuron(acts, "middle")
    except ValueError:
        pass
    else:
        raise AssertionError("bad mode should raise")

    print("selftest ok")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", nargs="?", help="text file, one input per line")
    ap.add_argument("-o", "--out", help="write TSV here instead of stdout")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--one-indexed", action="store_true", help="treat --layer as 1-based")
    ap.add_argument("--mode", default="all,last", help="comma-separated: all,last")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prepend-bos", action="store_true",
                    help="prepend <|endoftext|> so the first real token has context")
    ap.add_argument("--skip-first-position", action="store_true",
                    help="exclude position 0 from the max (drops the context-free artefact)")
    ap.add_argument("--hubs", default="1888,1790,2874", help="flag these as hubs; empty to disable")
    ap.add_argument("--strip", action="store_true", help="strip trailing whitespace instead of refusing")
    ap.add_argument("--allow-trailing-ws", action="store_true", help="keep trailing whitespace as-is")
    ap.add_argument("--md", action="store_true", help="also print a two-column markdown card")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.script:
        ap.error("script file required (or use --selftest)")

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            ap.error(f"--mode must be from {MODES}")

    hubs = {int(x) for x in args.hubs.split(",") if x.strip()} if args.hubs else set()

    lines = read_lines(args.script, strip=args.strip)
    if not lines:
        ap.error("no non-blank lines found")

    bad = find_trailing_ws(lines)
    if bad and not (args.strip or args.allow_trailing_ws):
        print("Refusing to run: trailing whitespace changes the destination.", file=sys.stderr)
        print("A trailing space is part of the key. Fix these, or pass", file=sys.stderr)
        print("--strip (remove it) or --allow-trailing-ws (keep it deliberately).\n", file=sys.stderr)
        for n, t in bad:
            print(f"  line {n}: {t!r}", file=sys.stderr)
        raise SystemExit(2)

    r = Resolver(args.model, args.layer, args.one_indexed, args.device,
                 prepend_bos=args.prepend_bos, skip_first=args.skip_first_position)
    meta = r.meta()
    print("# " + json.dumps(meta), file=sys.stderr)

    header = ["n", "tokens"] + [f"neuron_{m}" for m in modes]
    if hubs:
        header.append("hub")
    header.append("text")

    rows = []
    for i, text in enumerate(lines, 1):
        res, n_tok = r.resolve(text, modes)
        row = [str(i), str(n_tok)] + [str(res[m]) for m in modes]
        if hubs:
            row.append("HUB" if any(res[m] in hubs for m in modes) else "")
        row.append(text.replace("\t", "\\t"))
        rows.append(row)

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        print("# " + json.dumps(meta), file=out)
        print("\t".join(header), file=out)
        for row in rows:
            print("\t".join(row), file=out)
    finally:
        if args.out:
            out.close()

    if args.md:
        primary = modes[0]
        pi = 2 + modes.index(primary)
        print("\n| line | " + primary + " | text |", file=sys.stderr)
        print("|---|---|---|", file=sys.stderr)
        for row in rows:
            mark = " **(hub)**" if hubs and row[-2] == "HUB" else ""
            print(f"| {row[0]} | {row[pi]}{mark} | {row[-1]} |", file=sys.stderr)

    if hubs:
        n_hub = sum(1 for row in rows if row[-2] == "HUB")
        print(f"\n{n_hub}/{len(rows)} lines land in a hub {sorted(hubs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
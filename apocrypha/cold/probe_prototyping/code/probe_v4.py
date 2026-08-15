"""
probe_v4.py — first probe on the shared module. Two blocks, both to TSV.

  A. incidental doors. v3's controls scattered across 281, 1235, 1202, 13,
     1790, 945 while the deliberate probes converged on 20. Minimise each
     and record what it took to get there.

  B. carrier screening. v3 gave, to four decimals,
         delta = A[t*,20]  -  A[1, incumbent]
     for every carrier tried. If that holds over a wider set, route
     generation for a bigram-gated neuron needs no forward pass beyond the
     one that established its peak range.

Writes results/probe_v4_doors_*.tsv and results/probe_v4_carriers_*.tsv.
"""

import znou_probe as zp

WATCH = [20, 1430, 1888, 2256]

# ------------------------------------------------------- A. incidental doors

DOORS = [
    "at robin",         # 281,  delta 1.0496 — fattest arrival seen so far
    "at Quibinaw",      # 1235
    "at Mackerel",      # 1202
    "at Mackle",        # 13
    "at basin",         # 1790
    "at goblin",        # 1790
    "cold gums",        # 945, from the §8 fixture table
    "at Mackinaw",      # 20, the reference
    "at napkin",        # 1888, a null for comparison
]

# Perturbations of the fattest one. Is 281 a ' rob'+'in' bigram like 20,
# or does it survive changes the Mackinaw door would not?
ROBIN = [
    "at robin", "the robin", "on robin", "a robin",
    "at robins", "at robin's", "at Robin", "at ROBIN",
    "at robi", "at rob", "at robby", "at robing",
    "robin", " robin",
]


def block_a():
    rec = zp.Recorder("probe_v4_doors")
    print("--- A. incidental doors ---")
    print(f"  {'string':<16}{'dest':>6}{'delta':>9}{'t*':>4}{'runner':>8}  minimal prefix")
    for s in DOORS + ROBIN:
        r = zp.measure(s, watch=WATCH, block="doors")
        rec.add(r)
        pre = r.get("prefix", "")
        note = f"{pre!r} -> {r.get('prefix_dest')}" if pre != "" else "(bos footprint)"
        print(f"  {s!r:<16}{r['dest']:>6}{r['delta']:>9.4f}"
              f"{r['t_star']:>4}{r['runner_up']:>8}  {note}")
    print()
    return rec


# ---------------------------------------------------- B. carrier screening

TAIL = " Mackinaw"

CARRIERS = [
    "at", "the", "on", "near", "was", "from", "of", "to", "The", "in",
    "xq", "a", "an", "by", "for", "with", "into", "upon", "beyond",
    "island", "fort", "lake", "north", "old", "great", "we", "he", "they",
    "sailed", "beside", "past", "toward", "under", "aboard", "left",
    "Michigan", "Boston", "Iceland", "whale", "ship", "!", "?", "-", "1",
]


def block_b():
    rec = zp.Recorder("probe_v4_carriers")
    print("--- B. carrier screening ---")
    print("  predicted delta = peak_20(joined) - peak_incumbent(carrier alone)")
    print(f"  {'carrier':<12}{'dest':>6}{'delta':>9}{'peak20':>9}"
          f"{'inc':>6}{'incval':>9}{'pred':>9}{'err':>9}")

    for c in CARRIERS:
        joined = c + TAIL
        rj = zp.measure(joined, watch=WATCH, block="carriers", carrier=c)

        # incumbent: what the carrier alone reaches, and its peak value
        ra = zp.measure(c, watch=WATCH, with_prefix=False,
                        block="carrier_alone", carrier=c)
        inc, inc_val = ra["dest"], ra["peak_val"]

        peak20 = rj["peak_20"]
        pred = peak20 - inc_val
        # positive predicted margin means 20 clears the incumbent
        actual = rj["delta"] if rj["dest"] == 20 else -rj["delta"]
        err = actual - pred

        rj["incumbent"] = inc
        rj["incumbent_val"] = inc_val
        rj["pred_delta"] = round(pred, 6)
        rj["signed_delta"] = round(actual, 6)
        rj["pred_err"] = round(err, 6)
        rj["pred_sign_ok"] = int((pred > 0) == (rj["dest"] == 20))

        rec.add(rj)
        rec.add(ra)

        print(f"  {c!r:<12}{rj['dest']:>6}{rj['delta']:>9.4f}{peak20:>9.4f}"
              f"{inc:>6}{inc_val:>9.4f}{pred:>9.4f}{err:>9.4f}")

    joined_rows = [r for r in rec.rows if r.get("block") == "carriers"]
    ok = sum(r["pred_sign_ok"] for r in joined_rows)
    worst = max(abs(r["pred_err"]) for r in joined_rows)
    p20 = [r["peak_20"] for r in joined_rows]
    print(f"\n  sign correct {ok}/{len(joined_rows)}   worst |err| {worst:.6f}")
    print(f"  peak_20 range {min(p20):.4f} .. {max(p20):.4f}")
    print(f"  reached 20: {sum(r['dest'] == 20 for r in joined_rows)}/{len(joined_rows)}\n")
    return rec


if __name__ == "__main__":
    a = block_a()
    b = block_b()
    a.write()
    b.write()

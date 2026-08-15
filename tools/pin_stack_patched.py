"""
pin_stack.py — collects every remaining <TO PIN> value for J-SPACE §7.

Offline: reads the local HF cache, downloads nothing.
Run inside the znou env, paste the output back into §7.

PATCH 2026-08-15, two changes, both diagnostic rather than behavioural:

  1. The theta fingerprint is now computed TWICE — once unpinned, as the
     published §7 value was, and once with revision=REV. §7 states that
     "gpt2" alone is a moving pointer and insufficient to define D, so the
     script computing the pin should not itself rely on it. The unpinned
     computation is retained rather than replaced: removing it would make
     §7's published digest unreproducible. If the two agree, the local
     cache resolves correctly and the published value stands. If they
     disagree, §7's value describes whichever revision the cache happened
     to hold and needs regenerating from the pinned line.

  2. The numeric-settings block reported torch's ambient defaults under a
     header claiming they were "as §7 requires them". They are not —
     cudnn.allow_tf32 defaults to True, which is what §7 exists to
     disable. The block now reports found-vs-required side by side and
     sets nothing, so it remains a report of the environment.

Nothing about the hashing procedure itself is changed.
"""

import hashlib, json, platform, sys

REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

print("=" * 62)
print("EXECUTION STACK")
print("=" * 62)
print(f"python           {sys.version.split()[0]}")
print(f"platform         {platform.platform()}")

import torch, transformers
print(f"torch            {torch.__version__}")
print(f"transformers     {transformers.__version__}")
print(f"cuda (torch)     {torch.version.cuda}")
print(f"cudnn            {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"gpu              {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute cap      {cap[0]}.{cap[1]}")
    try:                                            # driver version
        import subprocess
        drv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], text=True).strip()
        print(f"driver           {drv}")
    except Exception as e:
        print(f"driver           <run: nvidia-smi>  ({e})")
    try:                                            # cuBLAS
        v = torch.cuda.cublas_version() if hasattr(torch.cuda, "cublas_version") else None
        print(f"cublas           {v if v else '<not exposed by this torch build>'}")
    except Exception:
        print("cublas           <not exposed by this torch build>")

# --- PATCH 2: report as found, against what §7 requires. Sets nothing. -----
print("\nnumeric settings AS FOUND in this process (this script sets none):")
_settings = [
    ("allow_tf32 (matmul)", torch.backends.cuda.matmul.allow_tf32, False),
    ("allow_tf32 (cudnn)", torch.backends.cudnn.allow_tf32, False),
    ("float32_matmul_prec", torch.get_float32_matmul_precision(), "highest"),
]
for label, found, required in _settings:
    flag = "ok" if found == required else "DIFFERS FROM §7"
    print(f"  {label:22} {str(found):8}  §7 requires {str(required):8}  {flag}")
print("  a measurement script MUST set these explicitly; see §7.")

print("\n" + "=" * 62)
print("MODEL / TOKENIZER ARTIFACT  (local cache, no download)")
print("=" * 62)

from huggingface_hub import scan_cache_dir

found = False
n_revs = 0
for repo in scan_cache_dir().repos:
    if repo.repo_id not in ("gpt2", "openai-community/gpt2"):
        continue
    found = True
    for rev in repo.revisions:
        n_revs += 1
        print(f"\nrepo             {repo.repo_id}")
        print(f"revision         {rev.commit_hash}"
              + ("   <- §7 pin" if rev.commit_hash == REV else ""))
        print(f"cached at        {rev.snapshot_path}")
        for f in sorted(rev.files, key=lambda x: x.file_name):
            if not f.file_name.endswith((".safetensors", ".bin", ".json", ".txt")):
                continue
            h = hashlib.sha256(open(f.file_path, "rb").read()).hexdigest()
            size = f.size_on_disk / 1e6
            print(f"  {f.file_name:26} {size:8.2f} MB  {h}")

if not found:
    print("\nNo gpt2 repo in the HF cache — the model may be loading from a local")
    print("directory instead. Point the same hashing loop at that folder.")
elif n_revs > 1:
    print(f"\n  NOTE: {n_revs} revisions cached. An unpinned load resolves to one of")
    print("  them and the choice is not this script's to make. The two digests")
    print("  below are the check.")

print("\n" + "=" * 62)
print("THETA FINGERPRINT  (what is actually loaded — format independent)")
print("=" * 62)

from transformers import GPT2LMHeadModel


def theta_sha256(mdl):
    """Unchanged from the original: names, dtypes, shapes, bytes, sorted."""
    h = hashlib.sha256()
    for name, t in sorted(mdl.state_dict().items()):
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# as the published §7 value was computed: no revision
mdl_unpinned = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.float32).eval()
sha_unpinned = theta_sha256(mdl_unpinned)
n_params = sum(p.numel() for p in mdl_unpinned.parameters())
del mdl_unpinned

# as §7 requires D to be defined: revision pinned
mdl_pinned = GPT2LMHeadModel.from_pretrained("gpt2", revision=REV).to(torch.float32).eval()
sha_pinned = theta_sha256(mdl_pinned)
del mdl_pinned

print(f"sha256(theta)    {sha_unpinned}   <- unpinned, as §7 publishes it")
print(f"sha256(theta)    {sha_pinned}   <- revision={REV[:8]}…")
print(f"n_params         {n_params:,}")

if sha_unpinned == sha_pinned:
    print("\nAGREE. The local cache resolves 'gpt2' to the pinned revision, so")
    print("§7's published digest describes the pinned weights. Either line may")
    print("be quoted; the pinned one is the one that stays true elsewhere.")
else:
    print("\nDISAGREE. An unpinned load on this machine does NOT resolve to the")
    print("§7 revision. The pinned line is the correct value for §7 and the")
    print("published digest should be regenerated from it. Any measurement")
    print("produced by a script that loads unpinned is suspect — check")
    print("token_sweep.py and data_pipeline.py, which §7.2 already flags as")
    print("predating the configuration discipline.")

print("\nThis is the strongest single value: it identifies the parameters")
print("themselves, independent of file format, repo layout or download path.")
print("=" * 62)

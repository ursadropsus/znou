"""
pin_stack.py — collects every remaining <TO PIN> value for J-SPACE §7.

Offline: reads the local HF cache, downloads nothing.
Run inside the znou env, paste the output back into §7.
"""

import hashlib, json, platform, sys

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

print("\nnumeric settings as §7 requires them:")
print(f"  allow_tf32 (matmul)   {torch.backends.cuda.matmul.allow_tf32}")
print(f"  allow_tf32 (cudnn)    {torch.backends.cudnn.allow_tf32}")
print(f"  float32_matmul_prec   {torch.get_float32_matmul_precision()}")

print("\n" + "=" * 62)
print("MODEL / TOKENIZER ARTIFACT  (local cache, no download)")
print("=" * 62)

from huggingface_hub import scan_cache_dir

found = False
for repo in scan_cache_dir().repos:
    if repo.repo_id not in ("gpt2", "openai-community/gpt2"):
        continue
    found = True
    for rev in repo.revisions:
        print(f"\nrepo             {repo.repo_id}")
        print(f"revision         {rev.commit_hash}")
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

print("\n" + "=" * 62)
print("THETA FINGERPRINT  (what is actually loaded — format independent)")
print("=" * 62)

from transformers import GPT2LMHeadModel

mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.float32).eval()
h = hashlib.sha256()
for name, t in sorted(mdl.state_dict().items()):
    h.update(name.encode())
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
print(f"sha256(theta)    {h.hexdigest()}")
print(f"n_params         {sum(p.numel() for p in mdl.parameters()):,}")
print("\nThis is the strongest single value: it identifies the parameters")
print("themselves, independent of file format, repo layout or download path.")
print("=" * 62)

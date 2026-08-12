from huggingface_hub import HfApi, hf_hub_download
import hashlib, torch

repo = "openai-community/gpt2"          # "gpt2" redirects here
sha = HfApi().repo_info(repo).sha
print("revision:", sha)

for f in ["model.safetensors", "vocab.json", "merges.txt", "config.json"]:
    path = hf_hub_download(repo, f, revision=sha)
    print(f, hashlib.sha256(open(path, "rb").read()).hexdigest())

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("cudnn", torch.backends.cudnn.version())
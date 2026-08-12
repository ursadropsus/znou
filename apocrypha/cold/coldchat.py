#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLD_TEAMS = {
    "top8":     [1508, 234, 2393, 1203, 2424, 591, 2227, 1944],
    "top2":     [1508, 234],
    "spec":     [38, 1103, 2094],
    "full":     [1508, 234, 2393, 1203, 2424, 591, 2227, 1944, 38, 1103, 2094],
}

AMP = 5.0
LAYER = 5
TEAM = "full"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading GPT-2 small…")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model.eval()

# ============================================================
# HOOK SYSTEM
# ============================================================

current_handle = None

def build_hook(team_indices):
    def hook(module, inputs, output):
        hidden = output.clone()          # (batch, seq, 3072)
        boost = torch.zeros_like(hidden[:, -1, :])
        boost[:, team_indices] = AMP
        hidden[:, -1, :] += boost
        return hidden
    return hook

def install_hook():
    global current_handle

    if current_handle:
        current_handle.remove()
        current_handle = None

    team_indices = COLD_TEAMS[TEAM]

    current_handle = model.transformer.h[LAYER].mlp.c_fc.register_forward_hook(
        build_hook(team_indices)
    )

install_hook()

# ============================================================
# GENERATION (silent, greedy)
# ============================================================

def generate(prompt, steered=False):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    attention_mask = torch.ones_like(input_ids)

    # turn off hook for baseline
    if not steered and current_handle:
        current_handle.remove()

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=60,
        do_sample=False,
        num_beams=1
    )

    # restore hook
    if not steered:
        install_hook()

    return tokenizer.decode(out[0], skip_special_tokens=True)

# ============================================================
# CLI
# ============================================================

print("\n=== COLD GPT-2 CHAT ===")
print("Commands: /team, /amp, /layer, /quit")
print(f"Active: team={TEAM}, layer={LAYER}, amp={AMP}")
print("===========================================\n")

while True:
    try:
        msg = input("You: ").strip()
    except KeyboardInterrupt:
        print("\nExiting.")
        break

    if not msg:
        continue

    # commands
    if msg.startswith("/"):
        parts = msg.split()
        if parts[0] == "/quit":
            print("Goodbye.")
            break

        elif parts[0] == "/team" and len(parts) == 2:
            if parts[1] in COLD_TEAMS:
                TEAM = parts[1]
                install_hook()
            else:
                print("Teams:", list(COLD_TEAMS.keys()))

        elif parts[0] == "/amp" and len(parts) == 2:
            AMP = float(parts[1])
            install_hook()
            print(f"[amp -> {AMP}]")

        elif parts[0] == "/layer" and len(parts) == 2:
            LAYER = int(parts[1])
            install_hook()

        else:
            print("Commands: /team X, /amp X, /layer X, /quit")

        continue

    # A/B
    print("\n--- BASELINE ---")
    print(generate(msg, steered=False))

    print("\n--- COLD ---")
    print(generate(msg, steered=True))

    print("-------------------------------------------\n")

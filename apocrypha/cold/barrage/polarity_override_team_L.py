import torch
import experiment_runner
import config
import os
import datetime

# --- The Intervention Toolkit ---

INTERVENTION_LAYER = 5

def run_surgical_intervention_suite(
    prompts_to_test: list[str],
    teams_to_test: dict[str, list[int]],
    amplification_strength: float = -10.0,
    output_file=None
):
    """
    Runs a suite of causal intervention experiments, testing multiple teams of neurons on each prompt.
    """
    
    def log(message):
        print(message)
        if output_file:
            output_file.write(message + "\n")

    model = experiment_runner.model
    tokenizer = experiment_runner.tokenizer
    if not model or not tokenizer:
        raise RuntimeError("Model or tokenizer not loaded.")

    log(f"\n--- Running Surgical Intervention Suite ---")
    log(f"  > Testing {len(teams_to_test)} different neuron teams.")
    log(f"  > Amplification Strength: +{amplification_strength}")

    for prompt in prompts_to_test:
        log(f"\n=================================================")
        log(f">>> MASTER PROMPT: \"{prompt}\"")
        log(f"=================================================")

        input_ids = model.to_tokens(prompt).to(config.DEVICE)

        # --- 1. Baseline Run (run once per master prompt) ---
        log("\n  --- BASELINE (NO INTERVENTION) ---")
        with torch.no_grad():
            baseline_output_tokens = model.generate(input_ids, max_new_tokens=15, do_sample=False)
        baseline_text = tokenizer.decode(baseline_output_tokens[0])
        log(f"    > Baseline Output: \"{baseline_text}\"")

        # --- 2. Intervened Runs (loop through all teams) ---
        for team_name, team_neurons in teams_to_test.items():
            log(f"\n  --- INTERVENTION: '{team_name}' Team ({len(team_neurons)} neurons) ---")
            
            def steering_hook(activation_tensor, hook):
                activation_tensor[0, -1, team_neurons] += amplification_strength
                return activation_tensor
            hook_name = f"blocks.{INTERVENTION_LAYER}.mlp.hook_post"
            
            with torch.no_grad(), model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                intervened_output_tokens = model.generate(input_ids, max_new_tokens=15, do_sample=False)
            intervened_text = tokenizer.decode(intervened_output_tokens[0])
            log(f"    > Intervened Output: \"{intervened_text}\"")
            
            if baseline_text.strip() != intervened_text.strip():
                log("    > CONCLUSION: Causal alteration detected.")
            else:
                log("    > CONCLUSION: No change from baseline.")

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- File Setup ---
    output_dir = "constellations"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = os.path.join(output_dir, f"intervention_report_surgical_{timestamp}.txt")

    with open(report_filename, "w") as f:
        # --- The Four Intervention Teams for Our Experiment ---
        COLD_CONSTELLATION_TEAMS = {
            "Brute Force (Top 8)": [1508, 234, 2393, 1203, 2424, 591, 2227, 1944],
            "Minimalist Coordinators (Top 2)": [1508, 234],
            "Specialists Only": [38, 1103, 2094], # The Sentry, The Grammarian, and the "Frigid" specialist
            "Full Roster (Combined)": list(set([1508, 234, 2393, 1203, 2424, 591, 2227, 1944, 38, 1103, 2094]))
        }
        
        # The prompt suite remains the same as our last rigorous test
        PROMPT_SUITE = [
            # Polar Opposites
            "The desert sun is beating down, the air is",
            "The summer sun is fiery, the season is",
            # Aligned & Neutral
            "It was cold",
            "It was frigid",
            # Orthogonal / Metaphorical
            "He reached out for something and it felt",
            "The first line of the novel is"
            # Valencies
            "The world is",
            "There is",
            "I am",
            "It was",
            "You are",
            "Wet and",
            "I exist",
            "I'm happy and",
            "The speed is",
            "the atmosphere feels",
            "the lighting seems",
            "the conversation was",
            "the decision feels",
            "the room was",
            "the timing is",
            "the gesture felt",
            "the outcome seems",
            "the expression was",
            "the change feels",
            "some might say",
            "The air tastes",
            "The silence feels",
            "The pattern is",
            "The reaction was",
            "The path ahead seems",
            "The moment felt",
            "The texture is",
            "The night was",
            "The rhythm feels",
            "The tension is",
            "The margin is",
            "The feeling lingers",
            "The signal was",
            "The memory is",
            "The color feels",
            "The future seems",
            "The interval was",
            "The surface feels",
            "The idea is",
            "The distance is",
            "The frame feels",
            # Valencies 2
            "The gesture carries",
            "The pressure is",
            "The signal feels",
            "The drift is",
            "The whisper was",
            "The boundary feels",
            "The cycle is",
            "The aftermath feels",
            "The outline is",
            "The scent is",
            "The window feels",
            "The current is",
            "The echo was",
            "The trace is",
            "The horizon feels",
            "The interval feels",
            "The pulse is",
            "The movement was",
            "The question is",
            # Random Curiosities
            "Mellybean",
            "Is it safe?",
            "What the",
            "Fuck",
            "I feel like you're a bit of a"
        ]

        run_surgical_intervention_suite(PROMPT_SUITE, COLD_CONSTELLATION_TEAMS, output_file=f)

    print(f"\nSUCCESS: Full surgical intervention report saved to '{report_filename}'")
# inference.py
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from model import GPT

# ----------------------------
# Config
# ----------------------------
CHECKPOINT_PATH = "checkpoints/checkpoint_step_34100.pt"
MODEL_ARGS = {
    "vocab_size": None,  # filled after tokenizer load
    "n_embd": 768,
    "block_size": 768 * 3,
    "n_layer": 12,
    "n_head": 12,
    "dropout": 0.1
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)

# ----------------------------
# Tokenizer + model load
# ----------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

MODEL_ARGS["vocab_size"] = tokenizer.vocab_size
model = GPT(MODEL_ARGS["vocab_size"],
            MODEL_ARGS["n_embd"],
            MODEL_ARGS["block_size"],
            MODEL_ARGS["n_layer"],
            MODEL_ARGS["n_head"],
            MODEL_ARGS["dropout"]).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint {CHECKPOINT_PATH} on {DEVICE}")

# ----------------------------
# Sampling helpers
# ----------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    logits: 1D tensor, vocab sized
    top_k: keep only top_k tokens with highest logits (0 -> disabled)
    top_p: keep the smallest set of tokens with cumulative prob >= top_p (0.0 -> disabled)
    """
    logits = logits.clone()
    vocab_size = logits.size(0)

    # Top-k
    if top_k is not None and top_k > 0:
        kth_val = torch.topk(logits, top_k).values[-1]
        logits[logits < kth_val] = filter_value

    # Top-p (nucleus)
    if top_p is not None and 0.0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # tokens to remove (True = remove)
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift right to keep at least one
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def generate_text(model, tokenizer, prompt,
                  max_new_tokens=50,
                  temperature=0.8,
                  top_k=0,
                  top_p=0.0,
                  repetition_penalty=1.0,
                  do_sample=True,
                  device=None):
    """
    Generates text from the causal LM.
    - top_k=0 disables top-k
    - top_p=0.0 disables nucleus
    - repetition_penalty>1.0 penalizes previously generated tokens
    - do_sample=False => greedy
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    prev_tokens = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated)          # (1, seq_len, vocab)
        next_token_logits = logits[:, -1, :].squeeze(0)  # (vocab,)

        # repetition penalty
        if repetition_penalty != 1.0 and len(prev_tokens) > 0:
            for t in set(prev_tokens):
                # divide logits for repeated tokens (simple heuristic)
                next_token_logits[t] = next_token_logits[t] / repetition_penalty

        # apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / float(temperature)

        # filtering (top-k / top-p)
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        if do_sample:
            probs = F.softmax(filtered_logits, dim=-1)
            # numerical guard
            if torch.isnan(probs).any() or probs.sum() <= 0:
                probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # shape (1,1)
        else:
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True).unsqueeze(0)

        generated = torch.cat((generated, next_token), dim=1)
        prev_tokens.append(int(next_token.item()))

        # stop if EOS produced
        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

    out_ids = generated[0].tolist()
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# ----------------------------
# Presets and demo prompts
# ----------------------------
presets = {
    "conservative": {"temperature": 0.6, "top_k": 40, "top_p": 0.0, "repetition_penalty": 1.2, "do_sample": True},
    "nucleus": {"temperature": 0.8, "top_k": 0, "top_p": 0.9, "repetition_penalty": 1.1, "do_sample": True},
    "greedy": {"do_sample": False, "temperature": 1.0, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.0},
}

PROMPTS = [
    "Write a short clear paragraph describing why exercise improves mental health:\n\n",
    "Jack is from France and he speaks the language",
    "The history of the church in medieval Europe can be summarized as"
]

# ----------------------------
# Run samples
# ----------------------------
if __name__ == "__main__":
    for name, cfg in presets.items():
        print("\n" + "="*30)
        print("PRESET:", name)
        for prompt in PROMPTS:
            print("\nPROMPT:", prompt)
            out = generate_text(model, tokenizer, prompt,
                                max_new_tokens=120,
                                device=DEVICE,
                                temperature=cfg.get("temperature", 1.0),
                                top_k=cfg.get("top_k", 0),
                                top_p=cfg.get("top_p", 0.0),
                                repetition_penalty=cfg.get("repetition_penalty", 1.0),
                                do_sample=cfg.get("do_sample", True))
            print("OUTPUT:")
            print(out)
            print("-"*60)

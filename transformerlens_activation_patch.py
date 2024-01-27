import torch

from transformer_lens import HookedTransformer


# We turn automatic differentiation off, to save GPU memory
# as this notebook focuses on model inference not model training.
torch.set_grad_enabled(False)

from neel_plotly import imshow

import transformer_lens.patching as patching

model = HookedTransformer.from_pretrained("gpt2")

prompts = [
    'After John and Mary went to the store, Mary gave a bottle of milk to',
    'After John and Mary went to the store, John gave a bottle of milk to',
]
answers = [
    (' Mary', ' John'),
    (' John', ' Mary'),
]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))
    ], device=model.cfg.device
)
print("Answer token indices", answer_token_indices)


def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE)


print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

resid_pre_act_patch_results = patching.get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, ioi_metric)


imshow(
    resid_pre_act_patch_results,
    yaxis="Layer",
    xaxis="Position",
    x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
    title="resid_pre Activation Patching"
)

import plotly.express as px
from nnsight import LanguageModel, util
from nnsight.tracing.Proxy import Proxy

# Load gpt2
model = LanguageModel("gpt2", device_map="cpu")

clean_prompt = "<|endoftext|>After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "<|endoftext|>After John and Mary went to the store, John gave a bottle of milk to"

correct_index = model.tokenizer(" John")["input_ids"][0]
incorrect_index = model.tokenizer(" Mary")["input_ids"][0]

print(f"' John': {correct_index}")
print(f"' Mary': {incorrect_index}")

# Enter nnsight tracing context
with model.forward() as runner:

    # Clean run
    with runner.invoke(clean_prompt) as invoker:
        clean_tokens = invoker.input["input_ids"][0]

        # Get hidden states of all layers in the network.
        # We index the output at 0 because it's a tuple where the first index is the hidden state.
        # No need to call .save() as we don't need the values after the run, just within the experiment run.
        clean_hs = [model.transformer.h[layer_idx].output[0] for layer_idx in range(len(model.transformer.h))]

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]).save()

    # Corrupted run
    with runner.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index] - corrupted_logits[0, -1, incorrect_index]
        ).save()

    ioi_patching_results = []

    # Iterate through all the layers
    for layer_idx in range(len(model.transformer.h)):
        _ioi_patching_results = []

        # Iterate through all tokens
        for token_idx in range(len(clean_tokens)):

            # Patching corrupted run at given layer and token
            with runner.invoke(corrupted_prompt) as invoker:

                # Apply the patch from the clean hidden states to the corrupted hidden states.
                model.transformer.h[layer_idx].output[0].t[token_idx] = clean_hs[layer_idx].t[token_idx]

                patched_logits = model.lm_head.output

                patched_logit_diff = patched_logits[0, -1, correct_index] - patched_logits[0, -1, incorrect_index]

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

                _ioi_patching_results.append(patched_result.save())

        ioi_patching_results.append(_ioi_patching_results)


print(f"Clean logit difference: {clean_logit_diff.value:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff.value:.3f}")

ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value.item(), Proxy)

clean_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

fig = px.imshow(
    ioi_patching_results,
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    labels={"x": "Position", "y": "Layer"},
    x=token_labels,
    title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
)

fig.show()

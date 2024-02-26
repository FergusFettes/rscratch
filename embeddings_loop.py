from nnsight import LanguageModel
import torch
import gc


prompt = "The quick brown fox jumps over _ _ _ _ _ _ _"
# prompt = "A dog is a dog. A cat is a cat. A rat is a"

model = LanguageModel("gpt2")
prompt_tokens = model.tokenizer.encode(prompt)


def count_empty(full_wte_output):
    # Get the index of the first consecutive output
    for i in range(len(full_wte_output[0])):
        if torch.all(full_wte_output[0][i] == full_wte_output[0][i + 1]):
            break
    print(f"First consecutive output index: {i}")


standin = model.tokenizer.encode(" _")[0]
token_position = prompt_tokens.index(standin)
print(f"Stand-in token position: {token_position} in prompt length: {len(prompt_tokens)}")
out_pos = token_position - 1
outs = []
with model.trace(use_cache=False) as runner:
    with runner.invoke(prompt) as _:
        # Get the output before the first stand-in
        full_wte_output = model.transformer.wte.output.save()
        output = model.lm_head.output.t[out_pos].argmax(-1).save()
    outs.append(output)
    out_pos += 1

gc.collect()
for out in outs:
    print(model.tokenizer.decode(out))
count_empty(full_wte_output)

with model.trace(use_cache=False) as runner:
    with runner.invoke(prompt) as _:
        # Replace the stand-ins with output
        for i, token in enumerate(outs):
            model.transformer.wte.input[0][0].t[token_position + i] = token
        full_wte_output = model.transformer.wte.output.save()
        output = model.lm_head.output.t[out_pos].argmax(-1).save()
    outs.append(output)
    out_pos += 1

gc.collect()
for out in outs:
    print(model.tokenizer.decode(out))
count_empty(full_wte_output)

with model.trace(use_cache=False) as runner:
    with runner.invoke(prompt) as _:
        # Replace the stand-ins with output
        for i, token in enumerate(outs):
            model.transformer.wte.input[0][0].t[token_position + i] = token
        full_wte_output = model.transformer.wte.output.save()
        output = model.lm_head.output.t[out_pos].argmax(-1).save()
    outs.append(output)
    out_pos += 1

gc.collect()
for out in outs:
    print(model.tokenizer.decode(out))

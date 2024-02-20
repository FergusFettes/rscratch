from nnsight import LanguageModel
import torch


prompt = "A dog is a dog. A cat is a cat. A rat is a _ _ _ _ _ _"
# prompt = "A dog is a dog. A cat is a cat. A rat is a"

model = LanguageModel("gpt2")
prompt_tokens = model.tokenizer.encode(prompt)

standin = model.tokenizer.encode(" _")[0]
token_position = prompt_tokens.index(standin)

outs = torch.tensor([], dtype=torch.int64)
with model.forward() as runner:
    with runner.invoke(prompt) as _:
        # Get the output before the first standin
        output = model.lm_head.output.t[token_position - 1].argmax(-1).save()
    outs = torch.cat([outs, output]).save()

    with runner.invoke(prompt) as _:
        # Replace the first standin with output
        model.transformer.wte.input[0][0][0][token_position: token_position + len(outs)] = outs
        # Bump the token position
        token_position += 1
        output = model.lm_head.output.t[token_position - 1].argmax(-1).save()
    outs = torch.cat([outs, output]).save()

    with runner.invoke(prompt) as _:
        # Replace the second standin with output
        model.transformer.wte.input[0][0][0][token_position: token_position + len(outs)] = outs
        # Bump the token position
        token_position += 1
        output = model.lm_head.output.t[token_position - 1].argmax(-1).save()
    outs = torch.cat([outs, output]).save()

    with runner.invoke(prompt) as _:
        # Replace the second standin with output
        model.transformer.wte.input[0][0][0][token_position: token_position + len(outs)] = outs
        # Bump the token position
        token_position += 1
        output = model.lm_head.output.t[token_position - 1].argmax(-1).save()
    outs = torch.cat([outs, output]).save()

    with runner.invoke(prompt) as _:
        # Replace the second standin with output
        model.transformer.wte.input[0][0][0][token_position: token_position + len(outs)] = outs
        # Bump the token position
        token_position += 1
        output = model.lm_head.output.t[token_position - 1].argmax(-1).save()
    outs = torch.cat([outs, output]).save()

for out in outs:
    print(model.tokenizer.decode(out.value))

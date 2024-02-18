from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cpu')


def decoder(x):
    with model.invoke(" ") as _:
        return model.lm_head(model.transformer.ln_f(x)).save()


# Simple source
source_prompt = "a dog is a dog. a rat is a"
source_tokens = model.tokenizer.encode(source_prompt)
rat = model.tokenizer.encode(" rat")[0]

rat_id = source_tokens.index(rat)
layer_states = []
# Get the hidden state across all layers
with model.invoke(source_prompt) as invoker:
    for layer in range(len(model.transformer.h)):
        layer_states.append(model.transformer.h[layer].output[0].save())


# Patch inputs inside a prompt
target_prompt = "a dog is a dog. a cat is a"
target_tokens = model.tokenizer.encode(target_prompt)
cat = model.tokenizer.encode(" cat")[0]

cat_id = target_tokens.index(cat)


# Now, patch the cat token to the rat hidden state
with model.invoke(target_prompt) as invoker:
    model.transformer.h[3].output[0][:, cat_id, :] = layer_states[3].value[:, rat_id, :]
    # Decode the output using the model
    model_output = model.lm_head.output[0].save()


last_output = model_output.value.argmax(dim=-1)[-1].tolist()
print(model.tokenizer.decode(target_tokens + [last_output]))














# Now, try replacing the tokens one by one across all the layers,
# to see when we output 'rat' and when 'cat'
outputs = []
for layer in range(len(model.transformer.h)):
    for position in range(len(target_tokens)):
        with model.invoke(target_prompt) as invoker:
            model.transformer.h[layer].output[0][:, position, :] = layer_states[layer].value[:, position, :]
            outputs.append((model.lm_head.output[0].save(), layer, position))


results = []
for output, layer, position in outputs:
    token = output.value.argmax(dim=-1).tolist()
    result = model.tokenizer.decode(token)
    if "rat" in result:
        print(f"Layer {layer}, position {position} outputs {result}")

from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cpu')


def decoder(x):
    with model.invoke(" ") as _:
        return model.lm_head(model.transformer.ln_f(x)).save()


# Simple source
source_prompt = "a dog is a dog. a rat is a"
source_tokens = model.tokenizer.encode(source_prompt)


with model.invoke(source_prompt) as invoker:
    rat_hidden_state = model.transformer.h[-1].output[0].save()
    # Decode the output using the model
    model_output = model.lm_head.output[0].save()

last_output = model_output.value.argmax(dim=-1)[-1].tolist()
print(model.tokenizer.decode(source_tokens + [last_output]))

# Decode the output manually
manual_output = decoder(rat_hidden_state)

manual_last_output = manual_output.value[0].argmax(dim=-1)[-1].tolist()
print(model.tokenizer.decode(source_tokens + [manual_last_output]))


# Simple target
target_prompt = "a dog is a"
target_tokens = model.tokenizer.encode(target_prompt)

# Simple target patched
with model.invoke(target_prompt) as invoker:
    model.transformer.h[-1].output[0][:, -1, :] = rat_hidden_state.value[:, -1, :]
    # Decode the output using the model
    model_output = model.lm_head.output[0].save()


last_output = model_output.value.argmax(dim=-1)[-1].tolist()
print(model.tokenizer.decode(target_tokens + [last_output]))


# Target in multi-token generation
source_prompt = "a dog is a dog. a cat is a cat. a rat"
source_tokens = model.tokenizer.encode(source_prompt)


target_prompt = "a dog is a dog. a cat is a cat. a bat"
target_tokens = model.tokenizer.encode(target_prompt)

completions = []
for i in range(len(model.transformer.h)):
    with model.invoke(source_prompt) as invoker:
        rat_hidden_state = model.transformer.h[i].output[0].save()
    outputs = []
    with model.generate(max_new_tokens=3) as runner:
        with runner.invoke(target_prompt) as invoker:
            model.transformer.h[i].output[0][:, :, :] = rat_hidden_state.value[:, :, :]
            # decode the output using the model
            for generation in range(3):
                outputs.append(model.lm_head.output[0].save())
                invoker.next()
    output = outputs[0].value.argmax(dim=-1).tolist()
    decoded = [model.tokenizer.decode(token) for token in output]
    output = outputs[1].value.argmax(dim=-1).tolist()
    decoded.append(model.tokenizer.decode(output))
    output = outputs[2].value.argmax(dim=-1).tolist()
    decoded.append(model.tokenizer.decode(output))
    decoded_str = "".join(decoded)
    last_sentence = decoded_str.split('.')[-1]
    completions.append(last_sentence)
print(completions)

decoded.insert(0, ' ')
source_words = [model.tokenizer.decode(token) for token in source_tokens]
decoded[:len(source_words) - 2] = source_words[:len(source_words) - 2]

output = outputs[1].value.argmax(dim=-1).tolist()
decoded.append(model.tokenizer.decode(output))

output = outputs[2].value.argmax(dim=-1).tolist()
decoded.append(model.tokenizer.decode(output))

rats = decoded.count(' rat')
bats = decoded.count(' bat')

print(f"Rats: {rats}")
print(f"Bats: {bats}")

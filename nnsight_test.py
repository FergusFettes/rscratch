from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cpu')


def decoder(x):
    return model.lm_head(model.transformer.ln_f(x))


prompt = 'The Eiffel Tower is in the city of'

# with model.generate(max_new_tokens=30) as generator:
#     with generator.invoke(prompt) as invoker:
#         layer_output_1 = model.transformer.h[-1].output[0].save()   # [1, 10, 768] aka [batch_size, seq_len, hidden_size] (hidden size is the size of the embedding aka context size)
#         logits_1 = decoder(layer_output_1).save()                # [1, 10, 50257] aka [batch_size, seq_len, vocab_size] (the decoding has essentially done an unembedding)
#         invoker.next()
#         layer_output_2 = model.transformer.h[-1].output[0].save()
#         logits_2 = decoder(layer_output_2).save()


with model.invoke(prompt) as invoker:
    layer_output_1 = model.transformer.h[-1].output[0].save()
    logits_1 = decoder(layer_output_1).save()
    invoker.next()
    layer_output_2 = model.transformer.h[-1].output[0].save()
    logits_2 = decoder(layer_output_2).save()



out_1 = model.tokenizer.decode(logits_1.value[0, -1, :].argmax())
print(f"out_1: {out_1}")

# Index into the logits to get the last token
out_1 = model.tokenizer.decode(logits_1.value[]

out_2 = model.tokenizer.decode(logits_2.value.argmax(dim=-1))
print(f"out_2: {out_2}")

out_3 = model.tokenizer.decode(logits_3.value.argmax(dim=-1))
print(f"out_3: {out_3}")

out_text = model.tokenizer.decode(generator.output[0])
print(out_text)



# with model.generate(max_new_tokens=1) as generator:
#     with generator.invoke('The Eiffel Tower is in the city of') as invoker:
#         hidden_states_pre = model.transformer.h[-8].mlp.output.clone().save()
#
#         noise = (0.1**0.5) * torch.randn(hidden_states_pre.shape)
#
#         model.transformer.h[-8].mlp.output = hidden_states_pre + noise
#
#         hidden_states_pre = model.transformer.h[-2].mlp.output.clone().save()
#
#         noise = (0.1**0.5) * torch.randn(hidden_states_pre.shape)
#
#         model.transformer.h[-1].mlp.output = hidden_states_pre + noise
#
#         embeddings_1 = model.transformer.wte.output.save()
#
#
# print(embeddings_1.value)
# out_text = model.tokenizer.decode(generator.output[0])
# print(out_text)


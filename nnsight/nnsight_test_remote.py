from nnsight import LanguageModel

model = LanguageModel('gpt2-xl')


def decoder(x):
    return model.lm_head(model.transformer.ln_f(x))


prompt = 'The Eiffel Tower is in the city of'

with model.generate(remote=True, max_new_tokens=3) as runner:
    with runner.invoke(prompt) as invoker:
        outputs_1 = model.lm_head.output[0].save()
        invoker.next()
        outputs_2 = model.lm_head.output[0].save()


output = model.tokenizer.decode(runner.output[0])
print('The model predicts', output)


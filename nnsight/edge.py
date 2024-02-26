from nnsight import LanguageModel


model = LanguageModel("gpt2")


prompt = "A typical definition of X would be '"
with model.generate(max_new_tokens=10) as runner:
    with runner.invoke(prompt) as invoker:
        output = model.transformer.lm_head.output.save()
    if [0].ge(0):
        raise ValueError("The word embeddings are not as expected.")


print(output)

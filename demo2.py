from nnsight import LanguageModel
import re


prompt = "A typical definition of X would be '"


model = LanguageModel("gpt2")


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


tokens = model.tokenizer.encode(prompt)
try:
    x = model.tokenizer.encode(" X")
    token_position = tokens.index(x[0])
except ValueError:
    x = model.tokenizer.encode("X")
    token_position = tokens.index(x[0])

testword = "apple"
with model.forward() as runner:
    with runner.invoke(testword) as invoker:
        testword_embeddings = model.transformer.wte.output.t[0].save()


def loop(runner):
    with runner.invoke(prompt) as _:
        output = model.lm_head.output.t[-1]
        probs = output.softmax(-1).save()
    return probs


with model.forward() as runner:
    while True:
        probs = loop(runner)

        # Use the `ge` method to generate a tensor with the condition
        max_prob_ge_threshold = probs.ge(0.05).save()

        if max_prob_ge_threshold.any():
            topk_indices = probs.topk(10).indices
            print(model.tokenizer.decode(topk_indices))
        else:
            print("No word found")
            break


__import__('ipdb').set_trace()
print(probs.value.max())
print(max_prob_ge_threshold.value)

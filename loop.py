from nnsight import LanguageModel


prompt = "A typical definition of X would be '"

model = LanguageModel("gpt2")
prompt_tokens = model.tokenizer.encode(prompt)


outputs = []


def working_loop(prompt_tokens, depth=0):
    prompt = model.tokenizer.decode(prompt_tokens)
    with model.forward() as runner:
        with runner.invoke(prompt) as _:
            output = model.lm_head.output.t[-1].save()

    # Need some way of returning! In this case we return after some depth
    if depth > 3:
        return {"output": output, "prompt": prompt}

    # Get the topk tokens and loop over them
    topk = output.value.topk(3)
    for token in topk.indices[0]:
        outputs.append(working_loop(prompt_tokens + [token], depth + 1))

    # Return the output and prompt
    return {"output": output, "prompt": prompt}


working_loop(prompt_tokens)

for output in outputs:
    print(output["prompt"])


outputs = []


def broken_loop(prompt_tokens, runner, depth=0):
    prompt = model.tokenizer.decode(prompt_tokens)
    with runner.invoke(prompt) as _:
        output = model.lm_head.output.t[-1]

    # Need some way of returning! In this case we return after some depth
    if depth > 3:
        return {"output": output.save(), "prompt": prompt}

    # Get the topk tokens and loop over them
    topk = output.topk(3)
    for token in topk.indices[0]:
        outputs.append(broken_loop(prompt_tokens + [token], runner, depth + 1))
    return {"output": output.save(), "prompt": prompt}


with model.forward() as runner:
    broken_loop(prompt_tokens, runner)


for output in outputs:
    print(output["prompt"])


outputs = []


def broken_loop(prompt_tokens, runner, depth=0):
    prompt = model.tokenizer.decode(prompt_tokens)
    with runner.invoke(prompt) as _:
        output = model.lm_head.output.t[-1]

    # Need some way of returning! In this case we return after some depth
    if depth > 3:
        return {"output": output.save(), "prompt": prompt}

    # Get the topk tokens and loop over them
    topk = output.topk(3)
    for token in topk.indices[0]:
        outputs.append(broken_loop(prompt_tokens + [token], runner, depth + 1))
    return {"output": output.save(), "prompt": prompt}


with model.forward() as runner:
    broken_loop(prompt_tokens, runner)


for output in outputs:
    print(output["prompt"])

from nnsight import LanguageModel
import gc


prompt = "The quick brown fox jumps over _ _ _ _ _ _ _"

model = LanguageModel("gpt2")
prompt_tokens = model.tokenizer.encode(prompt)

standin = model.tokenizer.encode(" _")[0]
token_position = prompt_tokens.index(standin)

print(f"Stand-in token position: {token_position} in prompt length: {len(prompt_tokens)}")


def init():
    out_pos = token_position - 1
    outs = []
    with model.trace() as runner:
        with runner.invoke(prompt) as _:
            output = model.lm_head.output.t[out_pos].argmax(-1).save()
        outs.append(output)
        out_pos += 1
    return outs, out_pos


def loop(outs, out_pos, depth=0):
    if depth == 3:
        return
    with model.trace() as runner:
        with runner.invoke(prompt) as _:
            # Replace the stand-ins with output
            for i, token in enumerate(outs):
                model.transformer.wte.input[0][0].t[token_position + i] = token
            output = model.lm_head.output.t[out_pos].argmax(-1).save()
        outs.append(output)
        out_pos += 1
    for out in outs:
        print(model.tokenizer.decode(out))
    loop(outs, out_pos, depth + 1)


def loop_with_gc(outs, out_pos, depth=0):
    if depth == 3:
        return
    with model.trace() as runner:
        with runner.invoke(prompt) as _:
            # Replace the stand-ins with output
            for i, token in enumerate(outs):
                model.transformer.wte.input[0][0].t[token_position + i] = token
            output = model.lm_head.output.t[out_pos].argmax(-1).save()
        outs.append(output)
        out_pos += 1
    gc.collect()
    for out in outs:
        print(model.tokenizer.decode(out))
    loop_with_gc(outs, out_pos, depth + 1)


def loop_with_gc_and_runner(outs, out_pos, depth=0, runner=None):
    if depth == 3:
        return
    with runner.invoke(prompt) as _:
        # Replace the stand-ins with output
        for i, token in enumerate(outs):
            model.transformer.wte.input[0][0].t[token_position + i] = token
        output = model.lm_head.output.t[out_pos].argmax(-1).save()
    outs.append(output)
    out_pos += 1
    gc.collect()
    for out in outs:
        print(model.tokenizer.decode(out))
    loop_with_gc_and_runner(outs, out_pos, depth + 1, runner)


outs, out_pos = init()
try:
    loop(outs, out_pos)
except Exception as e:
    print("loop failed:", e)

outs, out_pos = init()
loop_with_gc(outs, out_pos)
print("loop passed!")

outs, out_pos = init()
try:
    with model.trace() as runner:
        loop_with_gc_and_runner(outs, out_pos, runner=runner)
except Exception as e:
    print("loop failed:", e)

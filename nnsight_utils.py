def last_token(logits):
    return logits.value.argmax(dim=-1)[-1].tolist()


def decoder(x):
    with model.invoke(" ") as _:
        return model.lm_head(model.transformer.ln_f(x)).save()



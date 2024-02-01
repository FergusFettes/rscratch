from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope


source_context = SourceContext(device="cpu")
target_context = TargetContext.from_source(source_context, max_new_tokens=1)
patchscope = Patchscope(source_context, target_context)

patchscope.source.prompt = "a dog is a dog. a cat is a"
patchscope.target.prompt = "a dog is a dog. a rat is a"
patchscope.get_position_and_layer()
patchscope.target.max_new_tokens = 1

patchscope.source_forward_pass()
decoded_state = patchscope.source_decoder(patchscope._source_hidden_state)
output = decoded_state.value[0].argmax(dim=-1)[-1].tolist()
assert "cat" in output

patchscope.target_forward_pass()
output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
decoded = patchscope.target_model.tokenizer.decode(output)
assert "cat" in decoded


patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
patchscope.target.prompt = "a dog is a dog. a rat is a rat. a frog"

source_len = len(patchscope.source_tokens)
target_len = len(patchscope.target_tokens)

# # Patch the last two words
# patchscope.source.position = range(source_len - 2, source_len)
# patchscope.target.position = range(target_len - 2, target_len)

patchscope.target.max_new_tokens = 3
patchscope.get_position_and_layer()
patchscope.run()

output = patchscope._target_outputs[0].value.argmax(dim=-1).tolist()
decoded = [patchscope.target_model.tokenizer.decode(token) for token in output]
decoded

output = patchscope._target_outputs[1].value.argmax(dim=-1).tolist()
decoded = patchscope.target_model.tokenizer.decode(output)
decoded

output = patchscope._target_outputs[2].value.argmax(dim=-1).tolist()
decoded = patchscope.target_model.tokenizer.decode(output)
decoded

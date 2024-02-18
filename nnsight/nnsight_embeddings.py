from nnsight import LanguageModel

model = LanguageModel('gpt2', device_map='cpu')

# I want to test the embeddings.

prompt = 'The Eiffel Tower is in the city of'
with model.invoke(prompt) as invoker:
    embeddings = model.transformer.wte.output[0].save()
    layer_input = model.transformer.h[0].input[0][0].save()


print(embeddings.value.shape)
print(layer_input.value.shape)


# Now do a cosine similarity between the embeddings and the input
import torch
cos = torch.nn.CosineSimilarity(dim=0)

print(cos(embeddings.value, layer_input.value))


import torch
from transformers import GPT2Model, GPT2Tokenizer

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Encode the prompt
prompt = 'The Eiffel Tower is in the city of'
inputs = tokenizer(prompt, return_tensors='pt')

# Get the embeddings [batch_size, sequence_length, hidden_size]
with torch.no_grad():
    embeddings_torch = model.wte(inputs.input_ids)


print(cos(embeddings.value, embeddings_torch[0, :, :]))

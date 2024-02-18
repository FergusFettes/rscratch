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
    inputs_embeds = model.wte(inputs.input_ids)
    # Do the positional encoding too

# Let's assume you want the last token's embedding and its input to the first layer
# Since GPT-2 is auto-regressive, each token's input to a layer is the previous layer's output (or the embedding for the first layer)
last_token_embedding = inputs_embeds[0, -1, :]

# Forward the input through the first GPT-2 block (if you want to consider more than just the embedding layer)
with torch.no_grad():
    layer_output = model.h[0](inputs_embeds, None)[0]  # Getting the output of the first block

last_token_layer_input = layer_output[0, -1, :]

# Compute cosine similarity between the last token's embedding and its input to the first layer
cos_sim = torch.nn.CosineSimilarity(dim=0)
cos_similarity = cos_sim(last_token_embedding, last_token_layer_input)

print(cos_similarity.item())  # Output: Cosine similarity as a scalar value

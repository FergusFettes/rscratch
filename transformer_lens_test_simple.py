from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import torch
from torch import Tensor
from jaxtyping import Float


torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
source_model = HookedTransformer.from_pretrained("gpt2")
target_model = HookedTransformer.from_pretrained("gpt2")

source_prompt = "Amazon's former CEO attended Oscars"
target_prompt = "cat->cat; 135->135; hello->hello; black->black; shoe->shoe; start->start; mean->mean; ?"
print('Source prompt:', source_prompt, source_model.to_str_tokens(source_prompt))
print('Source token:', source_model.to_str_tokens(source_prompt)[4])
source_position = 4

print('Target prompt:', target_prompt, target_model.to_str_tokens(target_prompt))
print('Target token:', target_model.to_str_tokens(target_prompt)[-1])
target_position = len(target_model.to_str_tokens(target_prompt)) - 1

source_layer = 0
target_layer = 0
print(f'Source layer: {source_layer}, Target layer: {target_layer}')

_, source_cache = source_model.run_with_cache(source_prompt)
source_cache = source_cache["resid_pre", source_layer]


def hook_fn(
    target_activations: Float[Tensor, '...'],
    hook: HookPoint
) -> Float[Tensor, '...']:
    target_activations[:, target_position, :] = source_cache[:, source_position, :]
    return target_activations


# Target logits have shape [batch_size, seq_len, vocab_size]
target_logits = target_model.run_with_hooks(
    target_prompt,
    return_type="logits",
    fwd_hooks=[
        (get_act_name("resid_pre", target_layer), hook_fn)
    ]
)

prediction = target_logits.argmax(dim=-1).squeeze()[:-1]
print('Target model output :', target_model.to_string(prediction))
print(target_model.to_str_tokens(target_model.to_string(prediction)))

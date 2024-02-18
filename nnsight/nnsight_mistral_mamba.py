from nnsight import LanguageModel
from nnsight.models import Mamba

mistral = LanguageModel('mistralai/Mistral-7B-v0.1', device_map='cpu')
mamba = Mamba('state-spaces/mamba-130m', device_map='cuda:0')

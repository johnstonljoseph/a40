from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.olmo3 import Olmo3Config, Olmo3ForCausalLM
import torch.nn as nn

from a40.quant import QuantLinearWithWeights
from a40.custom.activation import IdentityActivation


class MyOlmo3Config(Olmo3Config):
    model_type = "my_olmo3"


class MyOlmo3ForCausalLM(Olmo3ForCausalLM):
    """
    Variant of Olmo3ForCausalLM whose post-attention / post-MLP / final norms
    are replaced with WeightedRMSNorm so gamma can be fused into neighboring linears.
    """
    config_class = MyOlmo3Config

    def __init__(self, config):
        super().__init__(config)
        config.architectures = [self.__class__.__name__]

        for layer in self.model.layers:
            mlp = layer.mlp
            act_fn = IdentityActivation().to(
                device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype
            )
            act_fn.set_blend(1.0)
            mlp.act_fn = act_fn

            names = ["gate_proj", "up_proj", "down_proj"]
            for name in names:
                linear_to_replace = getattr(mlp, name)
                linear = QuantLinearWithWeights(
                    linear_to_replace.in_features,
                    linear_to_replace.out_features,
                )
                setattr(mlp, name, linear)
                linear.to(linear_to_replace.weight.device)


class MyOlmo3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QuantLinearWithWeights(self.hidden_size, self.intermediate_size)
        self.up_proj = QuantLinearWithWeights(self.hidden_size, self.intermediate_size)
        self.down_proj = QuantLinearWithWeights(self.intermediate_size, self.hidden_size)

    def forward(self, x):
        down_proj = self.down_proj(self.gate_proj(x) * self.up_proj(x))
        return down_proj


# Enable auto-loading of the quantized subclass via AutoModelForCausalLM/AutoConfig
AutoConfig.register(MyOlmo3Config.model_type, MyOlmo3Config, exist_ok=True)
AutoModelForCausalLM.register(MyOlmo3Config, MyOlmo3ForCausalLM, exist_ok=True)









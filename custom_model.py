from transformers.models.olmo3 import Olmo3ForCausalLM

from a40.utils import WeightedRMSNorm


def _wrap_norm(norm_module):
    if isinstance(norm_module, WeightedRMSNorm):
        return norm_module
    return WeightedRMSNorm(norm_module)


class MyOlmo3ForCausalLM(Olmo3ForCausalLM):
    """
    Variant of Olmo3ForCausalLM whose post-attention / post-MLP / final norms
    are replaced with WeightedRMSNorm so gamma can be fused into neighboring linears.
    """

    def __init__(self, config):
        super().__init__(config)
        config.architectures = [self.__class__.__name__]
        self._convert_rms_norms()

    def _convert_rms_norms(self):
        for layer in self.model.layers:
            layer.post_attention_layernorm = _wrap_norm(layer.post_attention_layernorm)
            layer.post_feedforward_layernorm = _wrap_norm(layer.post_feedforward_layernorm)
        self.model.norm = _wrap_norm(self.model.norm)

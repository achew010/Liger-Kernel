import time
from dataclasses import dataclass

import torch
import logging
import triton
import functools
from fms_acceleration_foak.fused_ops.unsloth_lora.swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

logger = logging.getLogger(__name__)

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper

def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

class FoakSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        ori_shape = a.shape
        c = swiglu_fg_kernel(a, b)
        ctx.save_for_backward(a, b)
        return c.view(*ori_shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        ori_shape = dc.shape
        n_cols = ori_shape[-1]
        a, b = ctx.saved_tensors
        a = a.view(-1, n_cols)
        b = b.view(-1, n_cols)
        swiglu_DWf_DW_dfg_kernel(dc, a, b)
        return a.view(*ori_shape), b.view(*ori_shape)

class FoakSwiGLUMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):

        return self.down_proj(
            FoakSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )

def apply_foak_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = True,
    fused_linear_cross_entropy: bool = False,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Foak kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused lienar cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.llama import modeling_llama
    from fms_acceleration_foak.kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
    from fms_acceleration_foak.kernels.unsloth.rms_layernorm import fast_rms_layernorm
    from fms_acceleration_foak.kernels.unsloth.rope_embedding import fast_rope_embedding
    if rope:
        modeling_llama.apply_rotary_pos_emb = fast_rope_embedding
    if rms_norm:
        modeling_llama.LlamaRMSNorm.forward = fast_rms_layernorm
    if swiglu:
        modeling_llama.LlamaMLP = FoakSwiGLUMLP
    if cross_entropy:
        modeling_llama.CrossEntropyLoss = FastCrossEntropyLoss
    if fused_linear_cross_entropy:
        logger.warn("FOAK has no LCE implemetation. Skipping...")
        # modeling_llama.LlamaForCausalLM.forward = lce_forward


# Model type corresponds to the keys defined in transformers/models/auto/modeling_auto.py
MODEL_TYPE_TO_APPLY_LIGER_FN = {
    "llama": apply_foak_kernel_to_llama,
}

def _apply_foak_kernel(model_type: str = "", **kwargs) -> None:
    """
    Applies Liger kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_liger_kernel_to_* function.
    """

    if not model_type:
        logger.info("Model type was not provided. No Liger kernels will be applied.")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_LIGER_FN.keys():
        logger.info(
            f"There are currently no Liger kernels supported for model type: {model_type}."
        )
        return

    logger.info(f"Applying Liger kernels for model type: {model_type}.")
    # Apply the default combination of liger kernels available for the model
    MODEL_TYPE_TO_APPLY_LIGER_FN[model_type](**kwargs)

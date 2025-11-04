import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

@param('sparse_finetune', (False, True))
@param('value_expansion', (1, 2))
@param('core_heads', (1, 2))
@param('layers_for_mem_init', (None, 12))
@param('score_activation', (nn.Identity(), nn.ReLU()))
def test_ultra_mem(
    sparse_finetune,
    value_expansion,
    core_heads,
    layers_for_mem_init,
    score_activation,
):
    from ultra_mem.ultra_mem import UltraMem

    mem = UltraMem(
        512,
        value_expansion = value_expansion,
        core_heads = core_heads,
        layers_for_mem_init = layers_for_mem_init,
        score_activation = score_activation
    )

    tokens = torch.randn(1, 1024, 512)

    trainable_sparse_mask = None
    if sparse_finetune:
        trainable_sparse_mask = torch.randint(0, 2, (core_heads, mem.num_virtual_mems,)).bool()

    out, mem_indices, aux_loss = mem(tokens, trainable_sparse_mask = trainable_sparse_mask)

    assert (
        out.shape == x.shape and
        mem_indices.shape == (core_heads, *x.shape[:-1], 32),
        aux_loss.numel() == 1
    )

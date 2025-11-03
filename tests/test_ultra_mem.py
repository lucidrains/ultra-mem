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

    x = torch.randn(1, 1024, 512)

    mem = UltraMem(
        512,
        value_expansion = value_expansion,
        core_heads = core_heads,
        layers_for_mem_init = layers_for_mem_init,
        score_activation = score_activation
    )

    trainable_sparse_mask = None
    if sparse_finetune:
        trainable_sparse_mask = torch.randint(0, 2, (mem.num_virtual_mems,)).bool()

    out, aux_loss = mem(x, trainable_sparse_mask = trainable_sparse_mask)

    assert out.shape == x.shape and aux_loss.numel() == 1

from math import sqrt

import torch
from torch import nn, tensor, randn, arange, zeros
import torch.nn.functional as F
from torch.nn import Linear, Identity, Sequential, Parameter, Module, ModuleList

from einx import add, multiply
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# einstein notation
# b - batch
# n - sequence
# m - memories
# d, e - feature dimension (from / to)
# i, j - row, col
# h - heads
# r - tucker decomposition rank
# iv - implicit value expansion

# helper function

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def align_dims_to(t, target):
    if t.ndim >= target.ndim:
        return t

    ones = (1,) * (target.ndim - t.ndim)
    shape = t.shape
    return t.reshape(*shape, *ones)

def scale_gradient(t, scale = 1.):
    # scales the gradient, controlling effective lr upstream

    if scale == 1.:
        return t

    scaled_t = t * scale
    return t.detach() - scaled_t + scaled_t.detach()

# classes

class UltraMem(Module):

    def __init__(
        self,
        dim,
        dim_out = None,
        dim_values = None,
        num_memories = 1_000_000,
        topk = 32,
        dim_queries_keys = 128,         # think this is what PKM uses
        core_rank = 2,                  # the tucker decomposition core rank
        core_heads = 2,                 # number of cores / heads
        core_aux_loss_margin = 0.15,
        aux_loss_weight = 0.1,
        score_activation = nn.ReLU(),
        value_expansion = 4,
        pre_query_causal_conv = True,
        query_conv_kernel_size = 5,
        qk_layernorm = True,
        prenorm = True,
        proj_out = None,
        layers_for_mem_init = None,       # the number of layers in the transformer, used for deriving the proposed variance for initializing the memories - memories will be init to 1e-2 std otherwise
        mem_init_std = None,
        mem_lr_scale = 1e1,               # the values / memories needed 10x the learning rate (~1e-3 compared with ~1e-4 base lr, this could be controlled without doing param groups with a trick)
        mem_decay_lr_over_steps = 20_000  # decay the memory learning rate from 10x to 1x over this amount of training steps
    ):
        super().__init__()

        # variables

        assert sqrt(num_memories).is_integer()
        num_keys = int(sqrt(num_memories))

        dim_out = default(dim_out, dim)
        dim_values = default(dim_values, dim // core_heads)

        # prenorm and queries

        self.prenorm = nn.RMSNorm(dim) if prenorm else Identity()

        assert is_odd(query_conv_kernel_size)

        self.pre_query_causal_conv = Sequential(
            Rearrange('b n d -> b d n'),
            nn.ZeroPad1d((query_conv_kernel_size - 1, 0)),
            nn.Conv1d(dim, dim, query_conv_kernel_size, groups = dim),
            Rearrange('b d n -> b n d')
        ) if pre_query_causal_conv else Identity()

        self.to_queries = Linear(dim, dim_queries_keys, bias = False)

        # memory related

        self.heads = core_heads
        self.rank = core_rank

        self.num_keys = num_keys
        self.topk = topk

        self.query_ln = nn.LayerNorm(dim_queries_keys, bias = False) if qk_layernorm else Identity()
        self.key_ln = nn.LayerNorm(dim_queries_keys, bias = False) if qk_layernorm else Identity()

        self.keys = Parameter(randn(2, core_rank, num_keys, dim_queries_keys) * 1e-2)

        # their tucker decomposed core is 2x2
        # learned e2e with an auxiliary loss

        self.core = Parameter(randn(core_heads, core_rank, core_rank) * 1e-2)

        # handle value expansion

        assert divisible_by(num_memories, value_expansion)
        num_virtual_mems = num_memories // value_expansion

        self.value_expansion = value_expansion
        self.value_expansion_proj = Parameter(randn(value_expansion, dim_values, dim_values) * 1e-2)

        batch_randperm = randn(num_virtual_mems, value_expansion).argsort(dim = -1)
        self.register_buffer('rand_proj_mapping', batch_randperm.flatten().long())

        # score activation - defaults to ReLU proposed by Csordas

        self.score_activation = score_activation

        # memories

        mem_init_var = 1e-4

        if exists(layers_for_mem_init):
            mem_init_var = value_expansion / (2 * topk * core_heads * layers_for_mem_init)

        self.num_virtual_mems = num_virtual_mems

        self.memories = Parameter(randn(core_heads, num_virtual_mems, dim_values) * sqrt(mem_init_var))

        self.register_buffer('head_arange', arange(core_heads), persistent = False)

        # memories lr

        self._mem_lr_scale = mem_lr_scale
        self.mem_decay_lr_over_steps = mem_decay_lr_over_steps

        self.register_buffer('step', tensor(0))

        # whether to have a projection from (head * dim_values) back to (dim)

        proj_out = default(proj_out, core_heads * dim_values != dim_out)
        self.combine_values_to_out = Linear(core_heads * dim_values, dim_out, bias = False) if proj_out else Identity()

        # auxiliary loss on the core

        self.aux_loss_weight = aux_loss_weight
        self.core_aux_loss_margin = core_aux_loss_margin

        self.register_buffer('zero', tensor(0.), persistent = False)

    def reset_step_(self):
        self.step.zero_()

    @property
    def device(self):
        return self.zero.device

    @property
    def mem_lr_scale(self):

        step = self.step.item()
        init_lr, step_end_decay = self._mem_lr_scale, self.mem_decay_lr_over_steps

        if step > step_end_decay:
            return 1.

        slope = (1. - init_lr) / (step_end_decay - step)
        return init_lr + slope * step

    def forward(
        self,
        tokens,
        trainable_sparse_mask = None, # bool[num_memories,]
        return_aux_loss = None
    ):

        tokens = self.prenorm(tokens)

        # svd

        u, s, t = torch.svd(self.core)

        u_vec = u[..., 0]
        t_vec = t[..., 0]

        # aux loss on singular values

        return_aux_loss = default(return_aux_loss, self.training)

        aux_loss = self.zero

        if return_aux_loss:
            non_first_singular_values = s[:, 1:]

            aux_loss = F.relu(non_first_singular_values - self.core_aux_loss_margin).pow(2).mean(dim = -1) # eq (12)

            aux_loss = aux_loss.sum() * self.aux_loss_weight

        # queries keys

        tokens = self.pre_query_causal_conv(tokens)
        queries = self.to_queries(tokens)

        keys = self.keys

        # query key layernorm for stability

        queries = self.query_ln(queries)
        keys = self.key_ln(keys)

        row_scores, col_scores = einsum(queries, self.keys, 'b n d, rc r m d -> rc b n m r')

        # tucker decompsed qk retrieval, following fig 4

        merged_row_scores = einsum(row_scores, u_vec, '... r, h r -> h ...')
        merged_col_scores = einsum(col_scores, t_vec, '... r, h r -> h ...')

        top_row_indices = merged_row_scores.topk(self.topk, dim = -1).indices
        top_col_indices = merged_col_scores.topk(self.topk, dim = -1).indices

        indices = add('... i, ... j -> ... (i j)', top_row_indices * self.num_keys, top_col_indices)

        # ready for filtered row / col scores

        top_row_indices, top_col_indices = tuple(repeat(t, '... -> ... r', r = self.rank) for t in (top_row_indices, top_col_indices))
        row_scores, col_scores = tuple(repeat(t, '... -> h ...', h = self.heads) for t in (row_scores, col_scores))

        filtered_row_scores = row_scores.gather(-2, top_row_indices)
        filtered_col_score = col_scores.gather(-2, top_col_indices)

        scores = einsum(filtered_row_scores, filtered_col_score, self.core, 'h ... i r1, h ... j r2, h r1 r2 -> h ... i j')
        scores = rearrange(scores, '... i j -> ... (i j)')

        # get final scores and memory indices - (head, batch, seq, sel mems)

        final_scores, top_merged_score_indices = scores.topk(self.topk, dim = -1)
        final_indices = indices.gather(-1, top_merged_score_indices)

        # they use non-competitive scores, corroborating Csordas et al. 2023

        final_scores = self.score_activation(final_scores)

        # fetch the memories, and also handle sparse finetuning

        head_arange = align_dims_to(self.head_arange, final_indices)
        memory_indices = final_indices // self.value_expansion

        memories = self.memories[head_arange, memory_indices]

        # change gradients to memories if needed

        if self.training:
            memories = scale_gradient(memories, self.mem_lr_scale)

        # sparse finetuning

        if exists(trainable_sparse_mask):
            assert len(trainable_sparse_mask) == self.num_virtual_mems

            grad_mask = trainable_sparse_mask[final_indices]

            masked_memories = grad_mask * memories
            memories = memories.detach() + masked_memories - masked_memories.detach()

        # multiply by the scores and aggregate

        if self.value_expansion > 1:
            # handle the implicit value expansion
            expert_indices = self.rand_proj_mapping[final_indices]
            expert_indices = repeat(expert_indices, '... -> ... d', d = memories.shape[-1])

            shape = list(expert_indices.shape)
            shape[-2] = self.value_expansion

            scaled_memories = multiply('... m d, ... m', memories, final_scores)

            pooled_values = zeros(shape, device = self.device).scatter(-2, expert_indices, scaled_memories)
            aggregated = einsum(pooled_values, self.value_expansion_proj, '... iv d, iv d e -> ... e')
        else:
            aggregated = einsum(memories, final_scores, '... m d, ... m -> ... d')

        # concat the MCS heads and maybe combine

        concatted_heads = rearrange(aggregated, 'h b n d -> b n (h d)')

        out = self.combine_values_to_out(concatted_heads)

        # increment step

        if self.training:
            self.step.add_(1)

        # returning

        return out, aux_loss

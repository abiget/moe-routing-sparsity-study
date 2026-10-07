from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from .base import MoEAdapter


class MixtralAdapter(MoEAdapter):
    def update_model_top_k(
        self, model: AutoModelForCausalLM, new_top_k: int | Sequence[int]
    ) -> AutoModelForCausalLM:
        """
        Update the top_k value for each layer in the model.

        Args:
            model (AutoModelForCausalLM): The original model.
            new_top_k (int | Sequence[int]): The new top_k value(s) to set for each layer.
                If an integer is provided, it will be applied to all layers.
                If a sequence is provided, it must contain one value for each layer.
        Returns:
            AutoModelForCausalLM: The model with updated top_k values.
        """
        layers = model.model.layers

        if isinstance(new_top_k, int):
            top_k_by_layer = [new_top_k] * len(layers)
        else:
            top_k_by_layer = list(new_top_k)
            if len(top_k_by_layer) != len(layers):
                raise ValueError(
                    "Per-layer top_k must contain one value for every model layer."
                )

        model_top_k = model.config.num_experts_per_tok
        if any(top_k not in range(1, model_top_k + 1) for top_k in top_k_by_layer):
            raise ValueError("Every top_k value must be between 1 and num_experts.")

        for layer, top_k in zip(layers, top_k_by_layer):
            original_moe = layer.mlp
            if isinstance(original_moe, ModifiedMixtralMoE):
                original_moe = original_moe.original_moe
            layer.mlp = ModifiedMixtralMoE(original_moe, top_k)
        return model


class ModifiedMixtralMoE(torch.nn.Module):
    def __init__(self, original_moe, new_top_k):
        super().__init__()
        self.original_moe = original_moe
        self.new_top_k = new_top_k

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return compute_modified_moe_output(
            self.original_moe, hidden_states, self.new_top_k
        )


def run_one_expert(experts, hidden_states, expert_idx):
    """
    Run the MLP for a single expert.

    Args:
        experts: The experts module containing the MLPs.
        hidden_states: The input hidden states of shape [batch_size, sequence_length, hidden_dim].
        expert_idx: The index of the expert to run.
    Returns:
        torch.Tensor: The output of the expert's MLP.
    """
    gate, up = F.linear(hidden_states, experts.gate_up_proj[expert_idx]).chunk(
        2, dim=-1
    )

    out = experts.act_fn(gate) * up
    out = F.linear(out, experts.down_proj[expert_idx])
    return out.to(hidden_states.dtype)


def compute_modified_moe_output(
    original_moe: torch.nn.Module, hidden_states: torch.Tensor, new_top_k: int
) -> torch.Tensor:
    """
    Compute the output of the modified MLP with a new top_k value.

    Args:
        original_moe (torch.nn.Module): The original MoE module.
        hidden_states (torch.Tensor): Input hidden states of shape [batch_size, sequence_length, hidden_dim].
        new_top_k (int): The new top_k value (1 or 2).
    Returns:
        torch.Tensor: The output of the modified MoE.
    """

    flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [B*S, H]
    _, routed_weights, routed_indices = original_moe.gate(flat_hidden_states)
    top_k_indices = routed_indices[:, :new_top_k]  # [B*S, new_top_k]
    top_k_weights = routed_weights[:, :new_top_k]  # [B*S, new_top_k]
    if new_top_k == 1:
        top_k_weights = torch.ones_like(top_k_weights)  # [B*S, 1]
    else:
        # [B*S, new_top_k]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    expert_mask = torch.nn.functional.one_hot(
        top_k_indices, num_classes=original_moe.gate.num_experts
    )
    expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, B*S]
    # [num_experts_with_hits, 1]
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    final_hidden_states = torch.zeros_like(flat_hidden_states)

    experts = original_moe.experts

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat_hidden_states[token_idx]  # [selected_tokens, hidden_dim]
        # [selected_tokens, 2 * intermediate_dim] -> [selected_tokens, intermediate_dim], [selected_tokens, intermediate_dim]
        gate, up = torch.nn.functional.linear(
            current_state, experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(gate) * up
        # [selected_tokens, hidden_dim]
        current_hidden_states = nn.functional.linear(
            current_hidden_states, experts.down_proj[expert_idx]
        )
        # [selected_tokens, hidden_dim] weight each token's contribution by its routing probability to the top-k expert: token1 [contribution * p1, contribution * p2, contribution * p3], token2 [contribution * p2, contribution * p3, contribution * p4], ... token n [contribution * p1, contribution * p2, contribution * p3]
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    # [batch_size * sequence_length, hidden_dim]
    return final_hidden_states.reshape(hidden_states.shape)

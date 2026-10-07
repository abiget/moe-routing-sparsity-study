import csv
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from moe_routing_sparsity.data.text_batches import load_batches_from_file
from moe_routing_sparsity.utils import load_model_and_tokenizer, set_dtype, set_seed


def replace_one_layer(
    model: nn.Module, layer_idx: int, mode: int
) -> tuple[nn.Module, object]:
    """
    Replace the MLP layer of a specific layer in the model with a CounterfactualMLP that
    simulates the effect of using only the top-k experts for that layer.

    Args:
        model (nn.Module): The model containing the layer to be replaced.
        layer_idx (int): The index of the layer to be replaced.
        mode (int): The mode for the CounterfactualMLP. 0 for top-0, 1 for top-1, 2 for top-2.

    Returns:
        tuple[nn.Module, object]: A tuple containing the modified layer and the original MLP
        layer, which can be used to restore the original layer later.
    """
    layer = model.model.layers[layer_idx]
    original_mlp = layer.mlp

    layer.mlp = CounterfactualMLP(original_mlp, mode)

    return layer, original_mlp


class CounterfactualMLP(nn.Module):
    """
    A wrapper around the original MLP layer that simulates the effect of using only the top-k experts for that layer.
    """

    def __init__(self, original_mlp: nn.Module, mode: int):
        super().__init__()
        self.orginal_mlp = original_mlp
        self.mode = mode

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return counterfactual_moe(self.orginal_mlp, hidden_states, self.mode)


@torch.no_grad()
def counterfactual_moe(
    original_mlp: nn.Module, hidden_states: torch.Tensor, mode: int
) -> torch.Tensor:
    """
    A forward pass through the original MLP layer, but only using the top-k experts as specified by the mode.

    Args:
        original_mlp (nn.Module): The original MLP layer.
        hidden_states (torch.Tensor): The input hidden states to the MLP layer.
        mode (int): The mode for the CounterfactualMLP. 0 for top-0, 1 for top-1, 2 for top-2.

    Returns:
        torch.Tensor: The output hidden states after passing through the modified MLP layer.
    """
    B, S, H = hidden_states.shape
    flat = hidden_states.reshape(-1, H)  # [B*S, H]

    _, top_k_weights, top_k_indices = original_mlp.gate(flat)

    if top_k_indices.shape[1] != 2:
        raise ValueError("Expected top_k_indices to have shape [B*S, 2]")

    if mode == 0:
        return torch.zeros_like(hidden_states)

    experts = original_mlp.experts

    if mode == 1:
        out = torch.zeros_like(flat)  # [B*S, H]
        top1 = top_k_indices[:, 0]

        for expert_idx_tensor in torch.unique(top1):
            expert_idx = int(expert_idx_tensor.item())

            token_idx = torch.nonzero(top1.eq(expert_idx), as_tuple=False).squeeze(-1)
            x = flat.index_select(0, token_idx)

            y = run_one_expert(experts, x, expert_idx)
            # [num_tokens_for_expert, hidden_size]
            out.index_copy_(0, token_idx, y)
        return out.reshape(B, S, H)

    elif mode == 2:
        out = torch.zeros_like(flat)

        for expert_idx_tensor in torch.unique(top_k_indices):
            expert_idx = int(expert_idx_tensor.item())

            # positions where the current expert is top-1
            token_idx = torch.nonzero(
                top_k_indices[:, 0].eq(expert_idx), as_tuple=False
            ).squeeze(-1)

            if token_idx.numel() > 0:
                x = flat.index_select(0, token_idx.squeeze(-1))

                y = run_one_expert(experts, x, expert_idx)

                weights = top_k_weights.index_select(0, token_idx)[:, 0:1].to(y.dtype)

                out.index_add_(0, token_idx, y * weights)

            # positions where the current expert is top-2
            token_idx = torch.nonzero(
                top_k_indices[:, 1].eq(expert_idx), as_tuple=False
            ).squeeze(-1)

            if token_idx.numel() > 0:
                x = flat.index_select(0, token_idx.squeeze(-1))

                y = run_one_expert(experts, x, expert_idx)

                weights = top_k_weights.index_select(0, token_idx)[:, 1:2].to(y.dtype)

                out.index_add_(0, token_idx, y * weights)

        return out.reshape(B, S, H)

    raise ValueError(f"Invalid mode: {mode}. Expected 0, 1, or 2.")


@torch.no_grad()
def run_one_expert(experts, hidden_states, expert_idx):
    """
    Run a single expert on the given hidden states.

    Args:
        experts: The experts module containing the expert parameters.
        hidden_states: The input hidden states to the expert.
        expert_idx: The index of the expert to run.

    Returns:
        torch.Tensor: The output hidden states after passing through the expert.
    """
    gate, up = F.linear(hidden_states, experts.gate_up_proj[expert_idx]).chunk(
        2, dim=-1
    )

    x = experts.act_fn(gate) * up
    x = F.linear(x, experts.down_proj[expert_idx])
    return x.to(hidden_states.dtype)


def tokenwise_kl(ref_logits, counterfactual_logits, temperature=1.0):
    """
    Compute the token-wise KL divergence between the reference logits (full two expert per token) and the counterfactual logits (zero, one, or two experts).

    Args:
        ref_logits (torch.Tensor): The reference logits from the full model.
        counterfactual_logits (torch.Tensor): The counterfactual logits from the modified model.
        temperature (float): The temperature for scaling the logits before computing KL divergence.

    Returns:
        torch.Tensor: The token-wise KL divergence.
    """

    ref = ref_logits.float() / temperature
    cf = counterfactual_logits.float() / temperature
    ref_probs = torch.softmax(ref, dim=-1)
    counterfactual_probs = torch.log_softmax(cf, dim=-1)

    return F.kl_div(
        counterfactual_probs, ref_probs, reduction="none", log_target=False
    ).sum(dim=-1) * (temperature**2)


def print_summary_statistics(rows, total_forward, num_layers):
    print("\nSummary statistics:")
    print("=" * 70)
    print(f"Total forward passes: {total_forward}")
    print("=" * 70)

    def mean(values):
        return sum(values) / max(len(values), 1)

    # overall mean KL divergences
    all_kl0 = [row["kl_0"] for row in rows]
    all_kl1 = [row["kl_1"] for row in rows]
    all_kl2 = [row["kl_2"] for row in rows]

    print(
        f"\n 0 experts: mean KL={mean(all_kl0):.6f}, max KL={max(all_kl0):.6f}"
        f"\n 1 expert: mean KL={mean(all_kl1):.6f}, max KL={max(all_kl1):.6f}"
        f"\n 2 experts: mean KL={mean(all_kl2):.6f}, max KL={max(all_kl2):.6f}"
    )

    # Layer mean KL divergences
    print("\n Per-layer:")
    print("layer | mean KL(0)  mean KL(1) mean KL(2)  " " gain (1) -> 2")
    print("-" * 70)

    for layer_idx in range(num_layers):
        layer_rows = [row for row in rows if row["layer"] == layer_idx]

        m0 = mean([row["kl_0"] for row in layer_rows])
        m1 = mean([row["kl_1"] for row in layer_rows])
        m2 = mean([row["kl_2"] for row in layer_rows])

        print(
            f"{layer_idx:5d} | "
            f"{m0:.6f} "
            f"{m1:.6f} "
            f"{m2:.6f} "
            f"{m1 - m2:10.6f}"
        )


def save_results_to_csv(rows, output_csv):
    """
    Save the diagnostics results to a CSV file.

    Args:
        rows (list): List of dictionaries containing the diagnostics results.
        output_csv (str): Path to the output CSV file.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = [
        "sequence",
        "position",
        "layer",
        "kl_0",
        "kl_1",
        "kl_2",
        "delta_1",
        "delta_2",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Csv file saved to: {output_path}")


def run_diagnostics(args):

    set_seed(42)

    device = torch.device(args.device)

    dtype = set_dtype(args.dtype, device)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.cache_dir, device, dtype
    )

    model.eval()

    metadata = {
        "model_name": args.model_name,
        "cache_dir": args.cache_dir,
        "text_file": args.text_file,
        "output_csv": args.output_csv,
        "seq_len": args.seq_len,
        "num_sequences": args.num_sequences,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "temperature": args.temperature,
        "num_layers": len(model.model.layers),
        "num_experts": model.config.num_local_experts,
        "top_k": model.config.num_experts_per_tok,
    }
    print("\nDiagnostics metadata:")
    print(metadata)

    assert (
        model.config.num_experts_per_tok == 2
    ), "This script only supports top-2 routing."

    num_layers = len(model.model.layers)

    rows = []

    total_forward = 0

    for batch_start, batch in load_batches_from_file(
        args.text_file, tokenizer, args.seq_len, args.batch_size, args.num_sequences
    ):
        batch = batch.to(device)
        attention_mask = torch.ones_like(batch, dtype=torch.long, device=device)

        print(
            f"\nBatch {batch_start // args.batch_size + 1} / {args.num_sequences // args.batch_size}:"
        )

        with torch.no_grad():
            baseline_output = model(
                input_ids=batch, attention_mask=attention_mask, use_cache=False
            )

        baseline_logits = baseline_output.logits.detach()

        # For each layer, run top0, top1, and top2 experts
        for layer_idx in range(num_layers):
            print(f"   layer {layer_idx:2d}: ", end=" ", flush=True)

            outputs: Dict[int, torch.Tensor] = {}

            for mode in (0, 1, 2):
                layer, orginal_mlp = replace_one_layer(model, layer_idx, mode)

                try:
                    with torch.no_grad():
                        out = model(
                            input_ids=batch,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )

                    outputs[mode] = out.logits.detach()
                    total_forward += 1

                finally:
                    # Restore the original MLP layer
                    layer.mlp = orginal_mlp

            # Compute KL divergences for each mode
            kl_div0 = tokenwise_kl(baseline_logits, outputs[0], args.temperature)
            kl_div1 = tokenwise_kl(baseline_logits, outputs[1], args.temperature)
            kl_div2 = tokenwise_kl(baseline_logits, outputs[2], args.temperature)

            # reduction in kl damage when using top-1 instead of top-0
            delta1 = kl_div0 - kl_div1
            # reduction in kl damage when using top-2 instead of top-1
            delta2 = kl_div1 - kl_div2

            # Sanity check: mode 2 should reproduce the baseline
            max_kl_div2 = float(kl_div2.max().item())

            print(
                f"mean KL(0)={kl_div0.mean().item():.6f}, "
                f"KL(1)={kl_div1.mean().item():.6f}, "
                f"KL(2)={kl_div2.mean().item():.6f}",
                flush=True,
            )

            if max_kl_div2 > 1e-3:
                print(
                    f"Warning: Maximum KL divergence for top-2 is {max_kl_div2:.6f} > 1e-3",
                    flush=True,
                )

            # save every token for every layer and every mode
            B, S = kl_div0.shape

            for b in range(B):
                for pos in range(S):
                    rows.append(
                        {
                            "sequence": batch_start + b,  # sequence index
                            "position": pos,  # token position in the sequence
                            "layer": layer_idx,
                            "kl_0": float(kl_div0[b, pos].item()),
                            "kl_1": float(kl_div1[b, pos].item()),
                            "kl_2": float(kl_div2[b, pos].item()),
                            "delta_1": float(delta1[b, pos].item()),
                            "delta_2": float(delta2[b, pos].item()),
                        }
                    )
            del outputs

        del baseline_output
        del baseline_logits

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_results_to_csv(rows, args.output_csv)

    print_summary_statistics(rows, total_forward, num_layers)

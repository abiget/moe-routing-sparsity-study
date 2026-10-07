import json
import time
from pathlib import Path
from typing import Sequence

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_routing_sparsity.architectures.mixtral import MixtralAdapter
from moe_routing_sparsity.utils import (
    load_model_and_tokenizer,
    set_dtype,
    set_seed,
    synchronize_cuda,
)

ADAPTERS = {
    "mixtral": MixtralAdapter,
}


def benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    device: torch.device,
    tasks: Sequence[str] | None = None,
):
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    if tasks is None:
        tasks = ["hellaswag"]

    task_manager = TaskManager()
    task_manager.load(tasks)

    return timed_evaluate(lm, tasks, task_manager, device)


def timed_evaluate(lm, tasks, task_manager, device):
    synchronize_cuda(device)
    start_time = time.perf_counter()

    result = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        task_manager=task_manager,
    )

    synchronize_cuda(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "metrics": result,
        "efficiency": {
            "elapsed_seconds": elapsed_seconds,
        },
    }


def test_token_generation(model, tokenizer, device, prompt="Hello, how are you?"):
    """
    Test the model's ability to generate text given a prompt.

    Args:
        model (AutoModelForCausalLM): The model to test.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        device (torch.device): The device to run the model on.
        prompt (str): The input prompt for text generation.
    Returns:
        str: The generated text.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def get_adapter(model):
    model_type = model.config.model_type

    try:
        adapter_class = ADAPTERS[model_type]
    except KeyError:
        raise ValueError(f"No adapter found for model type '{model_type}'.")

    return adapter_class()


def run_benchmark(args):
    set_seed(42)

    device = torch.device(args.device)

    dtype = set_dtype(args.dtype, device)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.cache_dir, device, dtype
    )

    benchmark_results = {
        "metadata": {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "device": str(device),
            "dtype": str(dtype),
            "num_layers": len(model.model.layers),
        },
        "runs": dict(),
    }

    tasks = [
        "hellaswag",
        "arc_easy",
        "piqa",
        "winogrande",
        "openbookqa",
        "lambada_openai",
    ]

    original_top_k = model.config.num_experts_per_tok
    top_ks = tuple(range(1, original_top_k + 1))

    # baseline benchmark with original top_k
    model.eval()
    benchmark_results["runs"]["baseline"] = benchmark(
        model, tokenizer, args.batch_size, device, tasks=tasks
    )

    for top_k in top_ks:
        adapter = get_adapter(model)
        model = adapter.update_model_top_k(model, new_top_k=top_k)
        model.eval()

        print(f"\nRunning benchmark for top-k={top_k}...")

        benchmark_results["runs"][f"top_k_{top_k}"] = benchmark(
            model, tokenizer, args.batch_size, device, tasks=tasks
        )

    save_benchmark_results(benchmark_results, args.output_path)


def save_benchmark_results(results, out_path):
    """Save benchmark results to a JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, default=handle_non_serializable, indent=2)

    print(f"\nBenchmark results saved to {out_path}")

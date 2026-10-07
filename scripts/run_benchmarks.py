import argparse
from pathlib import Path

from moe_routing_sparsity.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Diagnostics script")
    parser.add_argument(
        "--model_name",
        default="FlameF0X/TinyMoE-100m-2x8-retrained",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--cache_dir", default="./checkpoints", help="Hugging Face cache directory."
    )
    parser.add_argument(
        "--output_path",
        default="./results/benchmarks.json",
        help="Path to save benchmark results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing sequences.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run the diagnostics on (e.g., 'cuda' or 'cpu').",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Data type for the model.",
    )

    args = parser.parse_args()

    model_slug = args.model_name.replace("/", "_")
    args.cache_dir = str(Path(args.cache_dir) / model_slug)

    args.output_path = str(
        Path(f"{args.output_path}").with_name(f"benchmark_{model_slug}.json")
    )

    run_benchmark(args)


if __name__ == "__main__":
    main()

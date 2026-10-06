import argparse
from pathlib import Path

from moe_routing_sparsity.diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Diagnostics script")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Hugging Face model ID.",
    )

    parser.add_argument(
        "--cache_dir",
        default=f"./checkpoints",
        help="Hugging Face cache directory.",
    )

    parser.add_argument(
        "--text_file",
        required=True,
        help="Path to the text file containing input text for diagnostics.",
    )

    parser.add_argument(
        "--output_csv",
        default="./results/diagnostics_results.csv",
        help="Path to the output file for diagnostics results.",
    )

    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length for the model input."
    )

    parser.add_argument(
        "--num_sequences", type=int, default=100, help="Number of sequences to process."
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

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for scaling logits in KL divergence computation.",
    )

    args = parser.parse_args()

    model_slug = args.model_name.replace("/", "_")
    args.cache_dir = str(Path(args.cache_dir) / model_slug)

    args.output_csv = str(Path(f"{args.output_csv}").with_name(
        f"diagnostics_{model_slug}.csv"
    ))

    run_diagnostics(args)


if __name__ == "__main__":
    main()

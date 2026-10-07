import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_dtype(dtype_str: str, device: torch.device):
    """
    Set the data type for the model based on the provided string and device.

    Args:
        dtype_str (str): The string representing the desired data type ("auto", "float32", "float16", "bfloat16").
        device (torch.device): The device on which the model will run.

    Returns:
        torch.dtype: The corresponding torch data type.
    """
    if dtype_str == "float32":
        dtype = torch.float32
    elif dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        if device.type == "cuda":
            dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch.float32

    return dtype


def synchronize_cuda(device):
    """Synchronize CUDA operations to ensure accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_model_and_tokenizer(
    model_name: str, cache_dir: str, device: torch.device, dtype: torch.dtype
):
    """
    Load the model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): The directory to cache the model.
        device (torch.device): The device on which to load the model.
        dtype (torch.dtype): The data type for the model.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    load_dotenv()

    hf_token = os.getenv("HF_HUB_TOKEN")

    tokenizer_kwargs = {"cache_dir": cache_dir}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    model_kwargs = {"cache_dir": cache_dir, "dtype": dtype}
    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model.to(device), tokenizer

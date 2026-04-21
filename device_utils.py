import torch


def get_device(preferred: str = "auto") -> torch.device:
    preferred = (preferred or "auto").strip().lower()

    if preferred == "cpu":
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this system.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
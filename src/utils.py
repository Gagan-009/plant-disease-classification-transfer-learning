import os
import random
import numpy as np
import pandas as pd
import torch

def set_seed(seed: int = 42):
    """
    Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filepath: str, model: torch.nn.Module, class_to_idx: dict, optimizer=None, epoch: int = 0, extra_info: dict = None):
    """
    Saves model weights, class index mappings, and metadata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "classes": [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])],
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra_info": extra_info or {}
    }
    torch.save(payload, filepath)
    print(f"[Checkpoint] Successfully saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, device: torch.device = None):
    """
    Reloads weights into the model from a checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[Checkpoint] Successfully reloaded from {filepath}")
    return checkpoint


def log_experiment(ledger_path: str, record: dict):
    """
    Appends experiment run metadata and metrics into an experiment ledger CSV.
    """
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    df_new = pd.DataFrame([record])
    if os.path.exists(ledger_path):
        df_existing = pd.read_csv(ledger_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(ledger_path, index=False)
    print(f"[Ledger] Updated experiment ledger at {ledger_path}")

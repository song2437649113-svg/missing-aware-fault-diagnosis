"""Inputs of the two diagnosis branches: frozen MAE features and Welch log-power spectral densities."""
import numpy as np
import torch
from scipy.signal import welch

from .config import MAE_ARCH


def check_signal(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 2 or not len(x):
        raise ValueError("Signals must be a non-empty array of windows.")
    x = x.reshape(len(x), -1)
    if x.shape[1] != MAE_ARCH["seq_len"]:
        raise ValueError("Window length does not match the MAE configuration.")
    if not np.isfinite(x).all():
        raise ValueError("Signals contain non-finite values.")
    return x


@torch.inference_mode()
def extract_features(mae, x, cfg, batch_size=32):
    """Return (MAE features, log10 PSD) for complete or completed windows.

    MAE branch: all patches pass through the frozen encoder with their physical
    positions; the normalized tokens are averaged over time.
    Spectral branch: Welch PSD (Hann window), DC bin removed, log10 with a floor.
    """
    x = check_signal(x)
    device = next(mae.parameters()).device
    mae.eval()
    features = []

    for start in range(0, len(x), batch_size):
        v = torch.from_numpy(x[start:start + batch_size]).to(device)
        z = mae.patch_embed(
            v.reshape(len(v), mae.num_patches, mae.patch_size)
        )
        pos = torch.arange(
            mae.num_patches, device=device
        )[None].expand(len(v), -1)

        for block in mae.blocks:
            z = block(z, positions=pos)

        features.append(mae.norm(z).mean(1).cpu().numpy())

    _, psd = welch(
        x.astype(np.float64),
        fs=1.0,
        window="hann",
        nperseg=cfg["nperseg"],
        noverlap=cfg["noverlap"],
        detrend="constant",
        scaling="density",
        axis=-1,
    )

    return (
        np.concatenate(features).astype(np.float32),
        np.log10(
            np.maximum(psd[:, 1:], cfg["psd_floor"])
        ).astype(np.float32),
    )

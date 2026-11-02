"""Completion of missing segments with the frozen MAE. Observed samples are never modified."""
import numpy as np
import torch


@torch.inference_mode()
def mae_complete(mae, observed, patch_mask, batch_size=32):
    """Fill missing patches with the MAE.

    observed   : (n, seq_len) float32, NaN at every missing sample.
    patch_mask : (n, num_patches) bool, True for a missing patch.
    Only visible patches enter the encoder, with their physical indices;
    all visible samples are returned unchanged.
    """
    p = int(mae.patch_size)
    length = int(mae.num_patches)
    sample_mask = np.repeat(patch_mask, p, axis=1)

    if observed.shape != sample_mask.shape:
        raise ValueError("Signal and missing mask have different shapes.")
    if not np.isnan(observed[sample_mask]).all():
        raise ValueError("Missing samples must be NaN.")
    if not np.isfinite(observed[~sample_mask]).all():
        raise ValueError("Visible samples contain non-finite values.")

    counts = patch_mask.sum(1)
    if np.any(counts >= length):
        raise ValueError("At least one visible patch is required.")

    output = observed.copy()
    device = next(mae.parameters()).device
    mae.eval()

    # Group windows by the number of missing patches, so each window may have its own missing rate.
    for count in np.unique(counts):
        if count == 0:
            continue

        ids = np.flatnonzero(counts == count)
        keep = length - int(count)

        for start in range(0, len(ids), batch_size):
            ix = ids[start:start + batch_size]
            order = np.argsort(
                patch_mask[ix], axis=1, kind="stable"
            )

            visible_ids = torch.as_tensor(
                order[:, :keep].copy(), device=device
            )
            restore = torch.as_tensor(
                np.argsort(order, axis=1), device=device
            )

            v = torch.from_numpy(
                np.nan_to_num(observed[ix], nan=0.0)
            ).to(device)

            tokens = torch.gather(
                v.reshape(len(ix), length, p),
                1,
                visible_ids[..., None].expand(-1, -1, p),
            )

            z = mae.patch_embed(tokens)
            for block in mae.blocks:
                z = block(z, positions=visible_ids)

            pred = mae.forward_decoder(mae.norm(z), restore)
            pred = pred.reshape(len(ix), -1).float().cpu().numpy()

            output[ix] = np.where(
                sample_mask[ix], pred, observed[ix]
            )

    if not np.isfinite(output).all():
        raise RuntimeError("Completed signals contain non-finite values.")
    if not np.array_equal(
        output[~sample_mask], observed[~sample_mask]
    ):
        raise RuntimeError("Completion modified observed samples.")

    return output

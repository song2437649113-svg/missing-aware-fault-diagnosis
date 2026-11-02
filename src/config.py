"""Settings used in the paper, shared by all modules."""

SEQ_LEN = 4096          # samples per window
PATCH_SIZE = 16         # samples per MAE token
K_SHOT = 5              # labelled windows per class and operating domain for the diagnosis heads

# ===== MAE architecture and pretraining (random patch masking) =====
MAE_ARCH = dict(
    seq_len=SEQ_LEN, patch_size=PATCH_SIZE, embed_dim=128,
    depth=4, num_heads=4, dec_embed_dim=64, dec_depth=2,
)
PRETRAIN_CFG = dict(
    mask_ratio=0.60, batch_size=64, epochs=500, lr=3e-3,
    weight_decay=0.05, shape_weight=0.8, seed=2026,
)

# ===== Spectral branch (Welch power spectral density) =====
SPECTRUM_CFG = dict(nperseg=512, noverlap=256, psd_floor=1e-12)

# ===== Few-shot diagnosis and rejection =====
DIAG_CFG = dict(
    K=K_SHOT, selection="kcenter", split_seed=2026, cal_per_cell=10,
    seeds=[7, 17, 27, 37, 47], epochs=300, lr=1e-3, dropout=0.10,
    temperature=0.10, lambda_align=0.10, lambda_sep=0.05,
    margin=0.20, warmup=20, time_dim=MAE_ARCH["embed_dim"],
    spec_dim=SPECTRUM_CFG["nperseg"]//2, feature_dim=64,
    **SPECTRUM_CFG, knn_k=5, id_reject_budget=0.02,
    support_conflict_ratio=1.0, scale_floor=0.10, log_mad_floor=0.10,
)


def assert_mae_matches_config(mae):
    actual = dict(
        seq_len=int(mae.num_patches*mae.patch_size), patch_size=int(mae.patch_size),
        embed_dim=int(mae.patch_embed.out_features), depth=len(mae.blocks),
        num_heads=int(mae.blocks[0].attn.num_heads),
        dec_embed_dim=int(mae.decoder_embed.out_features), dec_depth=len(mae.decoder_blocks),
    )
    if actual != MAE_ARCH:
        raise RuntimeError(f"MAE architecture {actual} does not match MAE_ARCH {MAE_ARCH}.")

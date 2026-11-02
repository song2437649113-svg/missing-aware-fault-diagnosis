"""Index-aware masked autoencoder (MAE): rotary position encoding on physical patch indices."""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .config import MAE_ARCH, PRETRAIN_CFG, assert_mae_matches_config


# ================= Index-aware RoPE =================
def apply_rotary_pos_emb(q, k, positions):
    """
    Index-aware RoPE: queries and keys are rotated by the true physical patch
    indices (positions), so the time axis is preserved after masked patches are removed.
    q, k shape: (B, num_heads, seq_len, head_dim)
    positions shape: (B, seq_len)
    """
    B, L = positions.shape
    head_dim = q.shape[-1]

    # frequency base
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(q.device) / head_dim))

    # rotation angles from the physical indices
    freqs = torch.einsum('bi,j->bij', positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, L, head_dim)

    # broadcast to q, k: (B, 1, L, head_dim)
    cos = emb.cos().unsqueeze(1)
    sin = emb.sin().unsqueeze(1)

    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ================= Transformer components =================
class IndexAwareAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, positions):
        B, N, C = x.shape

        # standard multi-head layout (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # physical positions are passed to RoPE
        q, k = apply_rotary_pos_emb(q, k, positions)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class IndexAwareBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = IndexAwareAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.mlp(self.norm2(x))
        return x


# ================= Index-aware MAE =================
class MaskedAutoencoder1D(nn.Module):
    def __init__(self, seq_len=None, patch_size=None, embed_dim=None, depth=None,
                 num_heads=None, dec_embed_dim=None, dec_depth=None):
        super().__init__()
        supplied = dict(seq_len=seq_len, patch_size=patch_size, embed_dim=embed_dim,
                        depth=depth, num_heads=num_heads, dec_embed_dim=dec_embed_dim, dec_depth=dec_depth)
        arch = {k: MAE_ARCH[k] if v is None else v for k, v in supplied.items()}
        seq_len, patch_size, embed_dim, depth, num_heads, dec_embed_dim, dec_depth = (
            arch[k] for k in ["seq_len", "patch_size", "embed_dim", "depth", "num_heads", "dec_embed_dim", "dec_depth"])
        if patch_size < 2 or seq_len % patch_size:
            raise ValueError("seq_len must be a multiple of patch_size, and patch_size must be at least 2.")
        self.architecture = dict(arch)
        self.shape_weight = PRETRAIN_CFG["shape_weight"]
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # Encoder
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.blocks = nn.ModuleList([IndexAwareBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, dec_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        self.decoder_blocks = nn.ModuleList([IndexAwareBlock(dec_embed_dim, num_heads) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_embed_dim)
        self.decoder_pred = nn.Linear(dec_embed_dim, patch_size)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # also return the original physical indices of the kept patches
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        B = x.shape[0]
        x = x.reshape(B, self.num_patches, self.patch_size)
        x = self.patch_embed(x)

        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # the encoder sees the true physical indices of the visible patches
        for blk in self.blocks:
            x = blk(x, positions=ids_keep)

        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # the decoder sequence is back in full order, so positions are [0, 1, ..., L-1]
        B, L, _ = x_.shape
        full_positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        for blk in self.decoder_blocks:
            x_ = blk(x_, positions=full_positions)

        x_ = self.decoder_norm(x_)
        return self.decoder_pred(x_)

    def forward(self, imgs, mask_ratio=None):
        mask_ratio = PRETRAIN_CFG["mask_ratio"] if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        target = imgs.reshape(imgs.shape[0], self.num_patches, self.patch_size)

        # ============================================================
        # Reconstruction objective: absolute level + local waveform shape
        # ============================================================
        # 1. MSE on raw values keeps the absolute level, including the DC offset of the signal
        loss_mse = (pred - target) ** 2

        # 2. Shape loss on per-patch z-normalized values fits the local oscillation (impacts)
        target_mean = target.mean(dim=-1, keepdim=True)
        target_std = target.std(dim=-1, keepdim=True) + 1e-6
        target_norm = (target - target_mean) / target_std

        pred_mean = pred.mean(dim=-1, keepdim=True)
        pred_std = pred.std(dim=-1, keepdim=True) + 1e-6
        pred_norm = (pred - pred_mean) / pred_std

        loss_shape = (pred_norm - target_norm) ** 2

        loss_total = loss_mse + self.shape_weight * loss_shape
        # ============================================================

        # loss on masked patches only
        loss = loss_total.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask


def load_configured_mae(path, device=None):
    """Load pretrained MAE weights; refuses weights whose architecture differs from MAE_ARCH."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint.get("architecture")
    if saved is None or dict(saved) != MAE_ARCH:
        raise RuntimeError(f"Checkpoint architecture {saved} does not match MAE_ARCH {MAE_ARCH}.")
    model = MaskedAutoencoder1D(**saved).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.shape_weight = checkpoint.get("pretrain_config", {}).get("shape_weight", 0.8)
    assert_mae_matches_config(model)
    return model.eval(), checkpoint


# ================= Self-supervised pretraining =================
class UnlabeledTVNGDataset(Dataset):
    """Training windows of all operating domains (files xtra_{0,1,2}.npy / ytra_{0,1,2}.npy); label 6 is excluded."""

    def __init__(self, base_path):
        data_list = []
        for i in range(3):
            data_path = os.path.join(base_path, f"xtra_{i}.npy")
            label_path = os.path.join(base_path, f"ytra_{i}.npy")
            if os.path.exists(data_path) and os.path.exists(label_path):
                data = np.load(data_path)
                labels = np.load(label_path)
                # drop label 6 (held-out fault type)
                mask = labels != 6
                data_filtered = data[mask]
                data_list.append(data_filtered)

        if not data_list:
            raise FileNotFoundError("No .npy training data found.")

        self.data = np.concatenate(data_list, axis=0).astype(np.float32)
        print(f"Pretraining windows: {self.data.shape[0]}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


def pretrain(data_dir, save_path, device=None):
    """Pretrain the MAE by masked reconstruction with PRETRAIN_CFG and save the checkpoint."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(PRETRAIN_CFG["seed"])
    np.random.seed(PRETRAIN_CFG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PRETRAIN_CFG["seed"])
    dataset = UnlabeledTVNGDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=PRETRAIN_CFG["batch_size"], shuffle=True,
                            num_workers=0, drop_last=True)

    model = MaskedAutoencoder1D(**MAE_ARCH).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_CFG["lr"],
                            weight_decay=PRETRAIN_CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_CFG["epochs"])

    loss_history = []
    for epoch in range(PRETRAIN_CFG["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(batch_x, mask_ratio=PRETRAIN_CFG["mask_ratio"])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{PRETRAIN_CFG['epochs']}] | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        'epoch': PRETRAIN_CFG["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'embed_dim': MAE_ARCH["embed_dim"],
        'patch_size': MAE_ARCH["patch_size"],
        'architecture': dict(MAE_ARCH),
        'pretrain_config': copy.deepcopy(PRETRAIN_CFG)
    }, save_path)
    return model.eval(), np.array(loss_history)

"""Dual-branch fusion heads, cross-domain class prototypes and the accept / review / OOD decision layer."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import extract_features
from .mae import MaskedAutoencoder1D


class FusionHead(nn.Module):
    """Fuses frozen MAE features and log-PSD into a unit-norm embedding."""

    def __init__(self, cfg):
        super().__init__()

        def branch(n):
            return nn.Sequential(
                nn.Linear(n, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
            )

        self.time = branch(cfg["time_dim"])
        self.spectrum = branch(cfg["spec_dim"])
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, cfg["feature_dim"]),
        )

    def forward(self, a, b):
        z = torch.cat(
            [self.time(a), self.spectrum(b)], dim=1
        )
        return F.normalize(self.fusion(z), dim=1)


class FaultDiagnosisUQ(nn.Module):
    """Frozen MAE + ensemble of fusion heads + reliability gates.

    `predict(signals, domains)` returns a dict; `decision` holds the predicted
    class (0-5) for accepted windows, -2 for windows sent for review and -1 for
    OOD candidates. `domains` is the operating domain (0, 1 or 2) of each window.
    """

    def __init__(
        self, mae, cfg, bank_n, support_n, aug_cfg=None
    ):
        super().__init__()
        self.mae = mae.requires_grad_(False)
        self.cfg = copy.deepcopy(cfg)
        self.aug_cfg = copy.deepcopy(aug_cfg)
        self.nt = cfg["time_dim"]
        self.ns = cfg["spec_dim"]

        self.heads = nn.ModuleList(
            [FusionHead(cfg) for _ in cfg["seeds"]]
        )

        for suffix, n in (("t", self.nt), ("s", self.ns)):
            buffers = {
                "fit_mean_": torch.zeros(n),
                "fit_scale_": torch.ones(n),
                "mean_": torch.zeros(3, n),
                "scale_": torch.ones(3, n),
                "bank_": torch.zeros(bank_n, n),
                "support_": torch.zeros(support_n, n),
            }
            for name, value in buffers.items():
                self.register_buffer(name + suffix, value)

        buffers = dict(
            prototypes=torch.zeros(
                len(cfg["seeds"]), 6, cfg["feature_dim"]
            ),
            bank_domain=torch.zeros(
                bank_n, dtype=torch.long
            ),
            bank_source=torch.zeros(
                bank_n, dtype=torch.long
            ),
            support_domain=torch.zeros(
                support_n, dtype=torch.long
            ),
            support_y=torch.zeros(
                support_n, dtype=torch.long
            ),
            radius=torch.ones(bank_n, 2),
            ood_loc=torch.zeros(
                3, 2, dtype=torch.float64
            ),
            ood_scale=torch.ones(
                3, 2, dtype=torch.float64
            ),
            risk_loc=torch.zeros(
                3, 2, dtype=torch.float64
            ),
            risk_scale=torch.ones(
                3, 2, dtype=torch.float64
            ),
            thresholds=torch.full(
                (2,), float("inf"), dtype=torch.float64
            ),
        )
        for name, value in buffers.items():
            self.register_buffer(name, value)

    def _domains(self, domains, n):
        d = np.asarray(domains)
        if d.ndim == 0:
            d = np.repeat(d, n)
        if (
            d.shape != (n,)
            or not np.isin(d, [0, 1, 2]).all()
        ):
            raise ValueError(
                "domains must be the operating domain 0/1/2 of each window."
            )
        return d.astype(np.int64)

    def _native(self, a, b, domains):
        device = self.bank_t.device
        a, b = [
            torch.as_tensor(
                v, dtype=torch.float32, device=device
            )
            for v in (a, b)
        ]
        d = torch.as_tensor(
            domains, dtype=torch.long, device=device
        )
        return (
            (a - self.mean_t[d]) / self.scale_t[d],
            (b - self.mean_s[d]) / self.scale_s[d],
            d,
        )

    @staticmethod
    def _knn(q, bank, k, query_ids=None, bank_ids=None):
        """
        The k neighbours must come from different source windows.
        With leave-source-out scoring, all copies of the query's source are excluded.
        """
        needed = k + int(query_ids is not None)
        if (
            bank_ids is None
            or len(torch.unique(bank_ids)) < needed
        ):
            raise ValueError("Not enough distinct reference windows.")

        vv, jj = [], []

        for start in range(0, len(q), 128):
            distance = (
                torch.cdist(q[start:start + 128], bank)
                / np.sqrt(q.shape[1])
            )

            if query_ids is not None:
                same_source = (
                    query_ids[start:start + 128, None]
                    == bank_ids[None]
                )
                distance.masked_fill_(
                    same_source, float("inf")
                )

            values, indices = [], []
            for _ in range(k):
                v, j = distance.min(1)
                values.append(v)
                indices.append(j)

                distance = distance.masked_fill(
                    bank_ids[None] == bank_ids[j, None],
                    float("inf"),
                )

            vv.append(torch.stack(values, dim=1))
            jj.append(torch.stack(indices, dim=1))

        values = torch.cat(vv)
        indices = torch.cat(jj)

        if not torch.isfinite(values).all():
            raise RuntimeError(
                "Invalid neighbour distances; check the number of reference windows."
            )

        return values, indices

    @torch.inference_mode()
    def _classify(self, a, b):
        device = self.bank_t.device
        a = (
            torch.as_tensor(
                a, dtype=torch.float32, device=device
            )
            - self.fit_mean_t
        ) / self.fit_scale_t

        b = (
            torch.as_tensor(
                b, dtype=torch.float32, device=device
            )
            - self.fit_mean_s
        ) / self.fit_scale_s

        embeddings = [head(a, b) for head in self.heads]
        logits = torch.stack([
            z @ self.prototypes[i].T
            for i, z in enumerate(embeddings)
        ])

        logp = F.log_softmax(
            logits.double() / self.cfg["temperature"],
            dim=-1,
        )
        member_p = logp.exp()
        p = member_p.mean(0)

        entropy = (
            -(p * p.clamp_min(1e-300).log()).sum(-1)
            / np.log(6)
        )
        mi = (
            entropy
            + (member_p * logp).sum(-1).mean(0) / np.log(6)
        ).clamp_min(0)

        return dict(
            probabilities=p.cpu().numpy(),
            predicted_class=p.argmax(1).cpu().numpy(),
            max_softmax=p.max(1).values.cpu().numpy(),
            entropy=entropy.cpu().numpy(),
            mutual_information=mi.cpu().numpy(),
            embedding_member0=embeddings[0].cpu().numpy(),
        )

    @torch.inference_mode()
    def _geometry(
        self, a, b, domains, predicted, query_ids=None
    ):
        qt, qs, d = self._native(a, b, domains)
        pred = torch.as_tensor(predicted, device=d.device)

        raw = torch.empty(len(d), 2, device=d.device)
        ratio = torch.empty(len(d), device=d.device)

        if query_ids is not None:
            query_ids = torch.as_tensor(
                query_ids, device=d.device
            )

        for dm in range(3):
            iq = torch.where(d == dm)[0]
            if not len(iq):
                continue

            ib = torch.where(self.bank_domain == dm)[0]
            sp = torch.where(self.support_domain == dm)[0]
            qids = None if query_ids is None else query_ids[iq]

            branches = (
                (qt[iq], self.bank_t[ib]),
                (qs[iq], self.bank_s[ib]),
            )

            for v, (q, bank) in enumerate(branches):
                distance, neighbors = self._knn(
                    q,
                    bank,
                    self.cfg["knn_k"],
                    qids,
                    self.bank_source[ib],
                )
                radii = self.radius[ib, v]
                floor = (
                    radii.median() * .05
                ).clamp_min(1e-8)

                local = radii[neighbors].median(1).values
                raw[iq, v] = (
                    distance[:, -1] / local.clamp_min(floor)
                )

            q = torch.cat(
                [
                    qt[iq] / np.sqrt(self.nt),
                    qs[iq] / np.sqrt(self.ns),
                ],
                dim=1,
            )
            s = torch.cat(
                [
                    self.support_t[sp] / np.sqrt(self.nt),
                    self.support_s[sp] / np.sqrt(self.ns),
                ],
                dim=1,
            )

            distance = torch.cdist(q, s) / np.sqrt(2)
            same = self.support_y[sp][None] == pred[iq, None]

            dp = distance.masked_fill(
                ~same, float("inf")
            ).min(1).values
            dn = distance.masked_fill(
                same, float("inf")
            ).min(1).values

            ratio[iq] = (dp + 1e-8) / (dn + 1e-8)

        return raw.cpu().numpy(), ratio.cpu().numpy()

    @torch.inference_mode()
    def predict_features(
        self, a, b, domains, query_ids=None
    ):
        self.eval()
        d = self._domains(domains, len(a))
        out = self._classify(a, b)

        raw, ratio = self._geometry(
            a, b, d, out["predicted_class"], query_ids
        )

        risk = np.column_stack([
            out["entropy"],
            out["mutual_information"],
        ])

        def buf(name):
            return getattr(self, name).detach().cpu().numpy()

        so = (
            (
                np.log(np.maximum(raw, 1e-12))
                - buf("ood_loc")[d]
            ) / buf("ood_scale")[d]
        ).max(1)

        sr = (
            (
                np.log(np.maximum(risk, 1e-12))
                - buf("risk_loc")[d]
            ) / buf("risk_scale")[d]
        ).max(1)

        eo = so - buf("thresholds")[0]
        er = sr - buf("thresholds")[1]

        ood = eo > 0
        alert = er > 0
        conflict = (
            ratio >= self.cfg["support_conflict_ratio"]
        )
        review = alert & conflict

        decision = out["predicted_class"].copy()
        decision[review] = -2
        decision[ood] = -1

        margin = np.minimum(
            er,
            np.log(
                np.maximum(ratio, 1e-12)
                / self.cfg["support_conflict_ratio"]
            ),
        )

        out.update(
            raw_ood=raw,
            raw_risk=risk,
            support_ratio=ratio,
            ood_score=so,
            review_score=sr,
            ood_excess=eo,
            review_excess=er,
            uncertainty_alert=alert,
            support_conflict=conflict,
            low_reliability=review,
            ood_candidate=ood,
            reject=ood | review,
            decision=decision,
            review_priority=np.maximum(eo, margin),
        )
        return out

    @torch.inference_mode()
    def predict(self, raw_signals, domains, batch_size=32):
        a, b = extract_features(
            self.mae, raw_signals, self.cfg, batch_size
        )
        return self.predict_features(a, b, domains)


def load_diagnosis_model(path, device=None):
    """Load the trained diagnosis model (frozen MAE, fusion heads, reference banks and thresholds)."""
    ckpt = torch.load(
        path, map_location="cpu", weights_only=True
    )
    if ckpt.get("protocol") != "missing_augmented_v1":
        raise ValueError(
            "This loader expects the missing-augmented diagnosis model."
        )

    model = FaultDiagnosisUQ(
        MaskedAutoencoder1D(**ckpt["architecture"]),
        ckpt["config"],
        ckpt["bank_n"],
        ckpt["support_n"],
        ckpt["augmentation"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)

    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model.to(device).eval()

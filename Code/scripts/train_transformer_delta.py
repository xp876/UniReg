"""DNA Transformer / DNABERT-style baseline for Δ regression.

This script is intentionally conservative and "optional":
- It requires `transformers` and `torch`.
- It can run fully offline if `--model_name_or_path` points to a local directory.

By default we freeze the encoder and train a small regression head to keep it
CPU-friendly. If you have GPU access, you can set --freeze_encoder no.

Outputs:
- nt_transformer_delta.test_predictions.tsv  (element_id, sequence, y_delta, pred_delta)
- nt_transformer_delta.metrics.json

Note: The model name is "nt_transformer_delta" for historical reasons.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as e:
    raise SystemExit("ERROR: torch not installed (need CPU PyTorch).") from e

try:
    # NOTE: Nucleotide Transformer v2 repositories expose an auto_map for MaskedLM.
    # AutoModel may fail with custom EsmConfig ("Unrecognized configuration class").
    # The official model card uses AutoModelForMaskedLM.
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ModuleNotFoundError as e:
    raise SystemExit(
        "ERROR: transformers is not installed.\n\n"
        "Install:\n"
        "  python -m pip install transformers sentencepiece\n\n"
        "If your cluster has no internet, pre-download the model and point --model_name_or_path to the local folder."
    ) from e

from common import compute_metrics, load_prepared_split, seed_everything, write_json, maybe_clip_quantile


class SeqDeltaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, delta_clip_q: float = 0.0):
        self.df = df.reset_index(drop=True).copy()
        y = pd.to_numeric(self.df["delta"], errors="coerce").values.astype(float)
        if delta_clip_q and delta_clip_q > 0:
            y = maybe_clip_quantile(y, float(delta_clip_q))
        self.df["delta"] = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return str(r["sequence"]), float(r["delta"])


def collate(batch, tokenizer, max_length: int):
    seqs, ys = zip(*batch)
    # many DNA models expect spaces between characters; we keep raw and let tokenizer handle
    toks = tokenizer(
        list(seqs),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    y = torch.tensor(ys, dtype=torch.float32)
    return toks, y, list(seqs)


class MeanPoolRegressor(nn.Module):
    def __init__(self, base, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, toks):
        # For MaskedLM models we pull token embeddings from the last hidden state.
        # Ensure hidden states are returned even if config defaults differ.
        out = self.base(**toks, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]  # (B, T, H)
        attn = toks.get("attention_mask")
        if attn is None:
            pooled = h.mean(dim=1)
        else:
            m = attn.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        pooled = self.drop(pooled)
        pred = self.head(pooled).squeeze(-1)
        return pred


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps, seqs = [], [], []
    for toks, y, seq in loader:
        toks = {k: v.to(device) for k, v in toks.items()}
        pred = model(toks)
        ys.append(y.numpy())
        ps.append(pred.cpu().numpy())
        seqs.extend(seq)
    return np.concatenate(ys), np.concatenate(ps), seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--model_name_or_path", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--cache_dir", default="", help="Optional HF cache dir (model/tokenizer files)")
    ap.add_argument("--revision", default="", help="Pin a specific HF revision/commit for reproducibility")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu (default) or cuda")
    ap.add_argument("--fp16", action="store_true", help="Use autocast fp16 on CUDA")
    ap.add_argument("--freeze_encoder", default="yes", choices=["yes", "no"], help="Freeze all encoder weights")
    ap.add_argument("--unfreeze_last_n", type=int, default=0, help="Unfreeze last N transformer blocks (works best on CUDA)")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--delta_clip_q", type=float, default=0.01)

    ap.add_argument("--offline", action="store_true", help="Set TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE and force local_files_only")
    ap.add_argument("--local_files_only", action="store_true", help="Force local_files_only even if not offline")
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.offline:
        import os
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    train_df = load_prepared_split(str(Path(args.prepared_dir) / "train.tsv"))
    val_df = load_prepared_split(str(Path(args.prepared_dir) / "val.tsv"))
    test_df = load_prepared_split(str(Path(args.prepared_dir) / "test.tsv"))

    for col in ["sequence", "delta", "element_id"]:
        if col not in train_df.columns:
            raise SystemExit(f"Missing required column '{col}' in prepared TSV")

    # drop NaNs
    train_df = train_df.dropna(subset=["sequence", "delta"]).copy()
    val_df = val_df.dropna(subset=["sequence", "delta"]).copy()
    test_df = test_df.dropna(subset=["sequence", "delta"]).copy()

    cache_dir = args.cache_dir or None
    revision = args.revision or None
    local_only = bool(args.offline or args.local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_only,
    )
    base = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_only,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    base.to(device)

    # --- freezing / partial unfreezing ---
    if args.freeze_encoder == "yes":
        for p in base.parameters():
            p.requires_grad = False

    def _get_blocks(m):
        # Try common transformer block containers.
        for path in [
            ("encoder", "layer"),      # BERT/RoBERTa
            ("encoder", "layers"),     # some encoders
            ("transformer", "h"),      # GPT2-like
            ("transformer", "layers"), # some decoders
            ("model", "layers"),       # some HF models
            ("layers",),               # last resort
        ]:
            cur = m
            ok = True
            for k in path:
                if not hasattr(cur, k):
                    ok = False
                    break
                cur = getattr(cur, k)
            if ok and hasattr(cur, "__len__"):
                try:
                    _ = len(cur)
                    return cur
                except Exception:
                    pass
        return None

    if int(args.unfreeze_last_n) > 0:
        blocks = _get_blocks(base)
        if blocks is None:
            print("[WARN] Could not locate transformer blocks to unfreeze; ignoring --unfreeze_last_n")
        else:
            n = int(args.unfreeze_last_n)
            n = min(n, len(blocks))
            for blk in list(blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

    # infer hidden size
    hidden = getattr(base.config, "hidden_size", None) or getattr(base.config, "d_model", None)
    if hidden is None:
        # fallback
        hidden = int(base.config.to_dict().get("hidden_size", 256))

    model = MeanPoolRegressor(base, hidden_size=int(hidden), dropout=float(args.dropout)).to(device)

    train_ds = SeqDeltaDataset(train_df, delta_clip_q=float(args.delta_clip_q))
    val_ds = SeqDeltaDataset(val_df, delta_clip_q=0.0)
    test_ds = SeqDeltaDataset(test_df, delta_clip_q=0.0)

    pin = bool(args.device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
                              collate_fn=lambda b: collate(b, tokenizer, args.max_length))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
                            collate_fn=lambda b: collate(b, tokenizer, args.max_length))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
                             collate_fn=lambda b: collate(b, tokenizer, args.max_length))

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction="mean", beta=0.5)

    best = -1e9
    best_path = out_dir / "nt_transformer_delta.best.pt"
    bad = 0

    use_amp = bool(args.fp16 and args.device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        for toks, y, _seq in train_loader:
            toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(toks)
                    loss = loss_fn(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(toks)
                loss = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        yv, pv, _ = predict_all(model, val_loader, device)
        sc = float(compute_metrics(yv, pv)["pearson"])
        if sc > best + 1e-6:
            best = sc
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= int(args.patience):
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    yt, pt, seqs = predict_all(model, test_loader, device)
    met = {
        "seed": int(args.seed),
        "val_best_pearson": float(best),
        "test": compute_metrics(yt, pt),
        "n_test": int(len(yt)),
        "model_name_or_path": str(args.model_name_or_path),
        "freeze_encoder": bool(args.freeze_encoder == "yes"),
        "unfreeze_last_n": int(args.unfreeze_last_n),
        "device": str(device),
        "fp16": bool(use_amp),
    }
    write_json(met, str(out_dir / "nt_transformer_delta.metrics.json"))

    pred_df = test_df[["element_id", "sequence"]].copy()
    pred_df["y_delta"] = yt
    pred_df["pred_delta"] = pt
    pred_df.to_csv(out_dir / "nt_transformer_delta.test_predictions.tsv", sep="\t", index=False)

    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()

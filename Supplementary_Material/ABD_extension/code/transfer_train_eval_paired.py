#!/usr/bin/env python3
"""Train the existing paired WT/MT/delta CNN on a source prepared_dir and evaluate on a target prepared_dir.

Incremental use-case:
- source: existing UniReg prepared split from out_plan8/prepared/split_seedX/weighted
- target: standardized HepG2 public variant-MPRA prepared split
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import compute_metrics, load_prepared_split, seed_everything, write_json
from models import SmallCNNMeanDelta, encode_seq, reverse_complement, derive_int_epi


class PairedDatasetWTMT(Dataset):
    def __init__(self, df: pd.DataFrame, rc_aug: bool = False, rc_prob: float = 0.5):
        self.seqs = df['sequence'].astype(str).tolist()
        self.y_wt = pd.to_numeric(df.get('log2_WT'), errors='coerce').values.astype(np.float32)
        self.y_mt = pd.to_numeric(df.get('log2_MT'), errors='coerce').values.astype(np.float32)
        self.y_delta = pd.to_numeric(df.get('delta'), errors='coerce').values.astype(np.float32)
        self.y_mean = pd.to_numeric(df.get('mean'), errors='coerce').values.astype(np.float32)
        sw = pd.to_numeric(df.get('sample_weight', 1.0), errors='coerce').fillna(1.0).values.astype(np.float32)
        self.w_wt = sw; self.w_mt = sw; self.w_delta = sw; self.w_mean = sw
        self.rc_aug = bool(rc_aug)
        self.rc_prob = float(rc_prob)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        if self.rc_aug and (np.random.rand() < self.rc_prob):
            s = reverse_complement(s)
        x = encode_seq(s)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_wt[idx]), torch.tensor(self.y_mt[idx]),
            torch.tensor(self.y_delta[idx]), torch.tensor(self.y_mean[idx]),
            torch.tensor(self.w_wt[idx]), torch.tensor(self.w_mt[idx]),
            torch.tensor(self.w_delta[idx]), torch.tensor(self.w_mean[idx]),
        )



def collate_paired_batch(batch):
    xs, wt, mt, delt, mean, w_wt, w_mt, w_delta, w_mean = zip(*batch)

    max_len = max(int(x.shape[0]) for x in xs)
    x_pad = torch.full((len(xs), max_len), 4, dtype=xs[0].dtype)  # 4 == N token
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0]] = x

    return (
        x_pad,
        torch.stack(wt),
        torch.stack(mt),
        torch.stack(delt),
        torch.stack(mean),
        torch.stack(w_wt),
        torch.stack(w_mt),
        torch.stack(w_delta),
        torch.stack(w_mean),
    )

def predict_all(model, loader, device):
    model.eval()
    y_wt=[]; y_mt=[]; y_delta=[]; y_mean=[]
    p_wt=[]; p_mt=[]; p_delta=[]; p_mean=[]
    with torch.no_grad():
        for (x, wt, mt, delt, mean, *_w) in loader:
            x = x.to(device)
            pm, pd = model(x)
            pi, pe = derive_int_epi(pm.cpu().numpy(), pd.cpu().numpy())
            y_wt.append(wt.numpy()); y_mt.append(mt.numpy())
            y_delta.append(delt.numpy()); y_mean.append(mean.numpy())
            p_wt.append(pi); p_mt.append(pe)
            p_delta.append(pd.cpu().numpy()); p_mean.append(pm.cpu().numpy())
    return {
        'y_wt': np.concatenate(y_wt), 'y_mt': np.concatenate(y_mt),
        'y_delta': np.concatenate(y_delta), 'y_mean': np.concatenate(y_mean),
        'p_wt': np.concatenate(p_wt), 'p_mt': np.concatenate(p_mt),
        'p_delta': np.concatenate(p_delta), 'p_mean': np.concatenate(p_mean),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_prepared', required=True)
    ap.add_argument('--target_prepared', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-2)
    ap.add_argument('--dropout', type=float, default=0.25)
    ap.add_argument('--loss', default='huber', choices=['huber','mse'])
    ap.add_argument('--huber_beta', type=float, default=0.5)
    ap.add_argument('--rc_aug', action='store_true')
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_train = load_prepared_split(Path(args.source_prepared) / 'train.tsv')
    src_val = load_prepared_split(Path(args.source_prepared) / 'val.tsv')
    src_test = load_prepared_split(Path(args.source_prepared) / 'test.tsv')
    tgt_test = load_prepared_split(Path(args.target_prepared) / 'test.tsv')

    device = torch.device('cpu')
    model = SmallCNNMeanDelta(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(reduction='none', beta=float(args.huber_beta)) if args.loss == 'huber' else nn.MSELoss(reduction='none')

    def mkloader(df, shuffle=False, rc=False):
        return DataLoader(
            PairedDatasetWTMT(df, rc_aug=rc),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_paired_batch,
        )

    train_loader = mkloader(src_train, shuffle=True, rc=args.rc_aug)
    val_loader = mkloader(src_val)
    src_test_loader = mkloader(src_test)
    tgt_test_loader = mkloader(tgt_test)

    best = -1e9
    bad = 0
    best_path = out_dir / 'transfer_paired_best.pt'

    for ep in range(1, args.epochs + 1):
        model.train()
        for (x, y_wt, y_mt, y_delta, y_mean, w_wt, w_mt, w_delta, w_mean) in train_loader:
            x = x.to(device)
            y_delta = y_delta.to(device)
            w_delta = w_delta.to(device)
            opt.zero_grad(set_to_none=True)
            pm, pd = model(x)
            loss = (loss_fn(pd, y_delta) * w_delta).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pack = predict_all(model, val_loader, device)
        val_score = float(compute_metrics(pack['y_delta'], pack['p_delta'])['pearson'])
        if val_score > best + 1e-6:
            best = val_score
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    src_pack = predict_all(model, src_test_loader, device)
    tgt_pack = predict_all(model, tgt_test_loader, device)
    src_m = compute_metrics(src_pack['y_delta'], src_pack['p_delta'])
    tgt_m = compute_metrics(tgt_pack['y_delta'], tgt_pack['p_delta'])

    src_pred = src_test[['element_id']].copy()
    src_pred['y_delta'] = src_pack['y_delta']
    src_pred['pred_delta'] = src_pack['p_delta']
    src_pred.to_csv(out_dir / 'source_test_predictions.tsv', sep='\t', index=False)

    tgt_pred = tgt_test[['element_id','variant_id']].copy() if 'variant_id' in tgt_test.columns else tgt_test[['element_id']].copy()
    tgt_pred['y_delta'] = tgt_pack['y_delta']
    tgt_pred['pred_delta'] = tgt_pack['p_delta']
    tgt_pred.to_csv(out_dir / 'target_test_predictions.tsv', sep='\t', index=False)

    out = {
        'seed': int(args.seed),
        'source_prepared': str(args.source_prepared),
        'target_prepared': str(args.target_prepared),
        'val_best_pearson': float(best),
        'source_test_pearson': float(src_m['pearson']),
        'source_test_spearman': float(src_m['spearman']),
        'target_test_pearson': float(tgt_m['pearson']),
        'target_test_spearman': float(tgt_m['spearman']),
        'n_target_test': int(len(tgt_pack['y_delta'])),
    }
    write_json(out, out_dir / 'transfer_paired.metrics.json')
    print(out)


if __name__ == '__main__':
    main()

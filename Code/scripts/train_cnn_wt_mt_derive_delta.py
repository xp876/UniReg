import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from common import (
    normalize_seq,
    set_seed,
    compute_metrics,
    write_json,
    maybe_clip_quantile,
)


def reverse_complement(seq: str) -> str:
    """Reverse-complement for DNA sequences (A/C/G/T/N)."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    s = normalize_seq(seq)
    return "".join(comp.get(c, "N") for c in s[::-1])


def one_hot_encode(seq: str) -> np.ndarray:
    s = normalize_seq(seq)
    arr = np.zeros((len(s), 4), dtype=np.float32)
    m = {'A':0,'C':1,'G':2,'T':3}
    for i, ch in enumerate(s):
        j = m.get(ch)
        if j is not None:
            arr[i, j] = 1.0
    return arr


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clip_q: float = 0.0, rc_aug: bool = False, rc_prob: float = 0.5):
        self.df = df.reset_index(drop=True)
        self.seqs = self.df['sequence'].astype(str).tolist()
        self.y_wt = pd.to_numeric(self.df['log2_WT'], errors='coerce').values.astype(np.float32)
        self.y_mt = pd.to_numeric(self.df['log2_MT'], errors='coerce').values.astype(np.float32)
        self.y_delta = pd.to_numeric(self.df['delta'], errors='coerce').values.astype(np.float32)
        self.w = pd.to_numeric(self.df.get('w_delta', 1.0), errors='coerce').fillna(1.0).values.astype(np.float32)
        if clip_q and clip_q > 0:
            self.y_delta = maybe_clip_quantile(self.y_delta, clip_q)

        self.rc_aug = bool(rc_aug)
        self.rc_prob = float(rc_prob)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        if self.rc_aug and (np.random.rand() < self.rc_prob):
            s = reverse_complement(s)
        x = one_hot_encode(s)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y_wt[idx]),
            torch.tensor(self.y_mt[idx]),
            torch.tensor(self.y_delta[idx]),
            torch.tensor(self.w[idx]),
        )


class EMA:
    """Simple exponential moving average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self._backup = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            self._backup[k] = sd[k].detach().clone()
            sd[k].copy_(v)

    def restore(self, model: nn.Module):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None


class PairCNN(nn.Module):
    def __init__(self, in_len: int, channels: int = 64, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(4, channels, kernel, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(channels, 2)

    def forward(self, x):
        # x: (B, L, 4)
        x = x.transpose(1, 2)  # (B, 4, L)
        h = self.net(x)
        h = h.mean(dim=-1)  # global average pool -> (B, C)
        out = self.head(h)  # (B,2)
        p_wt = out[:, 0]
        p_mt = out[:, 1]
        return p_wt, p_mt


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys_wt, ys_mt, ys_delta = [], [], []
    ps_wt, ps_mt = [], []
    ws = []
    for xb, ywt, ymt, ydel, w in loader:
        xb = xb.to(device)
        ywt = ywt.to(device)
        ymt = ymt.to(device)
        ydel = ydel.to(device)
        w = w.to(device)
        pwt, pmt = model(xb)
        ys_wt.append(ywt.cpu().numpy())
        ys_mt.append(ymt.cpu().numpy())
        ys_delta.append(ydel.cpu().numpy())
        ps_wt.append(pwt.cpu().numpy())
        ps_mt.append(pmt.cpu().numpy())
        ws.append(w.cpu().numpy())
    y_wt = np.concatenate(ys_wt)
    y_mt = np.concatenate(ys_mt)
    y_delta = np.concatenate(ys_delta)
    p_wt = np.concatenate(ps_wt)
    p_mt = np.concatenate(ps_mt)
    p_delta = p_wt - p_mt
    w = np.concatenate(ws)
    return dict(y_wt=y_wt, y_mt=y_mt, y_delta=y_delta, p_wt=p_wt, p_mt=p_mt, p_delta=p_delta, w=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--kernel', type=int, default=15)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--loss', default='huber', choices=['huber','mse'])
    ap.add_argument('--delta_clip_q', type=float, default=0.0)
    # compatibility flags used by the plan8 runner
    ap.add_argument('--rc_aug', action='store_true')
    ap.add_argument('--rc_prob', type=float, default=0.5)
    ap.add_argument('--use_ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--stop_on', default='delta', choices=['delta','wt','mt','avg_wt_mt'])
    ap.add_argument('--include_delta_loss', action='store_true', help='Also supervise Δ in addition to WT/MT (keeps Δ derived but adds extra loss).')
    ap.add_argument('--delta_loss_weight', type=float, default=0.5)
    args = ap.parse_args()

    set_seed(args.seed)

    prepared = Path(args.prepared_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(prepared / 'train.tsv', sep='\t')
    val_df = pd.read_csv(prepared / 'val.tsv', sep='\t')
    test_df = pd.read_csv(prepared / 'test.tsv', sep='\t')
    for df in (train_df, val_df, test_df):
        if 'seq' in df.columns and 'sequence' not in df.columns:
            df.rename(columns={'seq':'sequence'}, inplace=True)
        df['sequence'] = df['sequence'].astype(str).map(normalize_seq)

    L = len(train_df['sequence'].iloc[0])

    train_ds = SeqDataset(train_df, clip_q=args.delta_clip_q, rc_aug=args.rc_aug, rc_prob=args.rc_prob)
    val_ds = SeqDataset(val_df, clip_q=0.0)
    test_ds = SeqDataset(test_df, clip_q=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PairCNN(in_len=L, channels=args.channels, kernel=args.kernel, dropout=args.dropout).to(device)

    if args.loss == 'huber':
        crit = nn.SmoothL1Loss(reduction='none')
    else:
        crit = nn.MSELoss(reduction='none')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    best_val = -1e9
    best_path = out_dir / 'best.pt'

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, ywt, ymt, ydel, w in train_loader:
            xb = xb.to(device)
            ywt = ywt.to(device)
            ymt = ymt.to(device)
            ydel = ydel.to(device)
            w = w.to(device)

            pwt, pmt = model(xb)
            l_wt = crit(pwt, ywt)
            l_mt = crit(pmt, ymt)
            loss = (l_wt + l_mt) * w
            loss = loss.mean()

            if args.include_delta_loss:
                pdel = pwt - pmt
                l_del = crit(pdel, ydel) * w
                loss = loss + args.delta_loss_weight * l_del.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.item()))

        # validate (optionally with EMA weights)
        if ema is not None:
            ema.apply_to(model)
        pack = predict_all(model, val_loader, device)
        m_del = compute_metrics(pack['y_delta'], pack['p_delta'])
        m_wt = compute_metrics(pack['y_wt'], pack['p_wt'])
        m_mt = compute_metrics(pack['y_mt'], pack['p_mt'])
        val_delta = float(m_del.get('pearson', np.nan))
        val_wt = float(m_wt.get('pearson', np.nan))
        val_mt = float(m_mt.get('pearson', np.nan))

        if args.stop_on == 'wt':
            val_score = val_wt
        elif args.stop_on == 'mt':
            val_score = val_mt
        elif args.stop_on == 'avg_wt_mt':
            val_score = 0.5 * (val_wt + val_mt)
        else:
            val_score = val_delta

        if np.isfinite(val_score) and val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)

        if ema is not None:
            ema.restore(model)

        if ep % 5 == 0 or ep == 1:
            print(
                f"epoch {ep} train_loss={np.mean(losses):.4f} "
                f"val_delta={val_delta:.4f} val_wt={val_wt:.4f} val_mt={val_mt:.4f} "
                f"stop_on={args.stop_on} best={best_val:.4f}"
            )

    # load best and evaluate
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_pack = predict_all(model, val_loader, device)
    test_pack = predict_all(model, test_loader, device)

    metrics = {
        'val_delta': compute_metrics(val_pack['y_delta'], val_pack['p_delta']),
        'test_delta': compute_metrics(test_pack['y_delta'], test_pack['p_delta']),
        'val_wt': compute_metrics(val_pack['y_wt'], val_pack['p_wt']),
        'val_mt': compute_metrics(val_pack['y_mt'], val_pack['p_mt']),
        'test_wt': compute_metrics(test_pack['y_wt'], test_pack['p_wt']),
        'test_mt': compute_metrics(test_pack['y_mt'], test_pack['p_mt']),
        'config': vars(args),
    }
    write_json(metrics, out_dir / 'cnn_wt_mt_derive_delta.metrics.json')

    # write predictions
    # Use consistent naming with the rest of the pipeline: y_int/y_epi and pred_int/pred_epi.
    # Also keep legacy y_wt/y_mt + pred_wt/pred_mt columns for backward compatibility.
    vout = val_df[['element_id','sequence','log2_WT','log2_MT','delta']].copy()
    vout = vout.rename(columns={'log2_WT':'y_int','log2_MT':'y_epi','delta':'y_delta'})
    vout['pred_int'] = val_pack['p_wt']
    vout['pred_epi'] = val_pack['p_mt']
    vout['pred_delta'] = val_pack['p_delta']
    vout['y_wt'] = vout['y_int']
    vout['y_mt'] = vout['y_epi']
    vout['pred_wt'] = vout['pred_int']
    vout['pred_mt'] = vout['pred_epi']
    vout.to_csv(out_dir / 'cnn_wt_mt_derive_delta.val_predictions.tsv', sep='\t', index=False)

    tout = test_df[['element_id','sequence','log2_WT','log2_MT','delta']].copy()
    tout = tout.rename(columns={'log2_WT':'y_int','log2_MT':'y_epi','delta':'y_delta'})
    tout['pred_int'] = test_pack['p_wt']
    tout['pred_epi'] = test_pack['p_mt']
    tout['pred_delta'] = test_pack['p_delta']
    tout['y_wt'] = tout['y_int']
    tout['y_mt'] = tout['y_epi']
    tout['pred_wt'] = tout['pred_int']
    tout['pred_mt'] = tout['pred_epi']
    tout.to_csv(out_dir / 'cnn_wt_mt_derive_delta.test_predictions.tsv', sep='\t', index=False)

    print('Wrote:', out_dir)
    print('Best val Δ pearson:', best_val)
    print('Test Δ:', metrics['test_delta'])


if __name__ == '__main__':
    main()

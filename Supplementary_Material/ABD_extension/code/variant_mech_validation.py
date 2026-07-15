#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def safe_qcut(x, q=10):
    x = pd.Series(pd.to_numeric(x, errors='coerce'))
    ranks = x.rank(method='first')
    return pd.qcut(ranks, q=q, labels=False, duplicates='drop')


def write_fasta(df, seq_col, name_col, out_fa):
    out_fa = Path(out_fa)
    with open(out_fa, 'w', encoding='utf-8') as f:
        for _, r in df[[name_col, seq_col]].dropna().iterrows():
            nm = str(r[name_col]).replace(' ', '_')
            seq = str(r[seq_col]).upper()
            f.write(f'>{nm}\n{seq}\n')


def run_fimo(motif_db, fasta, out_tsv):
    cmd = ['fimo', '--verbosity', '1', '--text', motif_db, fasta]
    with open(out_tsv, 'w', encoding='utf-8') as out:
        subprocess.run(cmd, check=True, stdout=out)


def parse_fimo_hits(fimo_tsv, regexes):
    df = pd.read_csv(fimo_tsv, sep='\t', comment='#', low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=['family_regex','sequence_name','motif_id'])

    # normalize FIMO column names across versions
    original_cols = list(df.columns)
    norm_map = {}
    for c in original_cols:
        c_norm = str(c).strip().lower().replace(' ', '_').replace('-', '_')
        norm_map[c] = c_norm
    df = df.rename(columns=norm_map)

    motif_candidates = [c for c in ['motif_id', 'pattern_name', 'motif'] if c in df.columns]
    if motif_candidates:
        motif_col = motif_candidates[0]
    else:
        motif_col = df.columns[0]

    name_candidates = [c for c in ['sequence_name', 'sequence', 'seq_name'] if c in df.columns]
    if name_candidates:
        name_col = name_candidates[0]
    else:
        # FIMO text output is usually motif_id, motif_alt_id, sequence_name, ...
        # fall back to 3rd column if present, otherwise 2nd, otherwise first.
        if len(df.columns) >= 3:
            name_col = df.columns[2]
        elif len(df.columns) >= 2:
            name_col = df.columns[1]
        else:
            name_col = df.columns[0]

    rows = []
    for fam in regexes:
        mask = df[motif_col].astype(str).str.contains(fam, case=False, regex=True, na=False)
        sub = df.loc[mask, [motif_col, name_col]].drop_duplicates().copy()
        sub['family_regex'] = fam
        rows.append(sub.rename(columns={motif_col: 'motif_id', name_col: 'sequence_name'}))

    if not rows:
        return pd.DataFrame(columns=['family_regex','sequence_name','motif_id'])
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--std_tsv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--motif_db', default='')
    ap.add_argument('--run_fimo', type=int, default=1)
    ap.add_argument('--top_n', type=int, default=300)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    std = pd.read_csv(args.std_tsv, sep='\t')
    std['abs_delta'] = std['delta'].abs()
    if 'emvar' not in std.columns:
        std['emvar'] = np.where(std['fdr'].notna() & (std['fdr'] < 0.1), 1, 0)

    # Step 1: effect-size bins
    std['effect_bin'] = safe_qcut(std['abs_delta'], q=10)
    eff = std.groupby('effect_bin', dropna=False).agg(
        n=('variant_id','size'),
        mean_abs_delta=('abs_delta','mean'),
        emvar_rate=('emvar','mean'),
        median_fdr=('fdr','median')
    ).reset_index()
    eff.to_csv(out_dir / 'effectsize_bins.tsv', sep='\t', index=False)

    # Step 2: focus family list
    fam_txt = out_dir / 'focus_family_regexes.txt'
    subprocess.run(['python', str(Path(__file__).resolve().parent / 'extract_liver_focus_families.py'), '--root', str(root), '--out_txt', str(fam_txt)], check=True)
    regexes = [x.strip() for x in fam_txt.read_text(encoding='utf-8').splitlines() if x.strip()]

    summary_rows = []
    summary_rows.append({'metric':'n_variants', 'value': len(std)})
    summary_rows.append({'metric':'n_emvar', 'value': int(pd.to_numeric(std['emvar'], errors='coerce').fillna(0).astype(int).sum())})
    summary_rows.append({'metric':'mean_abs_delta', 'value': float(std['abs_delta'].mean())})

    # Step 3: optional FIMO on strongest cases and background
    if args.run_fimo and args.motif_db:
        top = std.sort_values('abs_delta', ascending=False).head(args.top_n).copy()
        bg = std.sort_values('abs_delta', ascending=True).head(min(args.top_n, len(std))).copy()
        top['name_ref'] = top['variant_id'].astype(str) + '|ref|top'
        top['name_alt'] = top['variant_id'].astype(str) + '|alt|top'
        bg['name_ref'] = bg['variant_id'].astype(str) + '|ref|bg'
        bg['name_alt'] = bg['variant_id'].astype(str) + '|alt|bg'

        top_ref_fa = out_dir / 'top_ref.fa'
        top_alt_fa = out_dir / 'top_alt.fa'
        bg_ref_fa = out_dir / 'bg_ref.fa'
        bg_alt_fa = out_dir / 'bg_alt.fa'
        write_fasta(top, 'sequence_ref', 'name_ref', top_ref_fa)
        write_fasta(top, 'sequence_alt', 'name_alt', top_alt_fa)
        write_fasta(bg, 'sequence_ref', 'name_ref', bg_ref_fa)
        write_fasta(bg, 'sequence_alt', 'name_alt', bg_alt_fa)

        top_ref_tsv = out_dir / 'fimo_top_ref.tsv'
        top_alt_tsv = out_dir / 'fimo_top_alt.tsv'
        bg_ref_tsv = out_dir / 'fimo_bg_ref.tsv'
        bg_alt_tsv = out_dir / 'fimo_bg_alt.tsv'
        run_fimo(args.motif_db, str(top_ref_fa), str(top_ref_tsv))
        run_fimo(args.motif_db, str(top_alt_fa), str(top_alt_tsv))
        run_fimo(args.motif_db, str(bg_ref_fa), str(bg_ref_tsv))
        run_fimo(args.motif_db, str(bg_alt_fa), str(bg_alt_tsv))

        def family_count(tsv):
            hits = parse_fimo_hits(tsv, regexes)
            if hits.empty:
                return pd.DataFrame({'family_regex': regexes, 'n_hit_sequences': 0})
            cnt = hits.groupby('family_regex')['sequence_name'].nunique().reset_index(name='n_hit_sequences')
            return cnt

        tr = family_count(top_ref_tsv).rename(columns={'n_hit_sequences':'top_ref_hits'})
        ta = family_count(top_alt_tsv).rename(columns={'n_hit_sequences':'top_alt_hits'})
        br = family_count(bg_ref_tsv).rename(columns={'n_hit_sequences':'bg_ref_hits'})
        ba = family_count(bg_alt_tsv).rename(columns={'n_hit_sequences':'bg_alt_hits'})
        summ = tr.merge(ta, on='family_regex', how='outer').merge(br, on='family_regex', how='outer').merge(ba, on='family_regex', how='outer').fillna(0)
        summ['top_total_hits'] = summ['top_ref_hits'] + summ['top_alt_hits']
        summ['bg_total_hits'] = summ['bg_ref_hits'] + summ['bg_alt_hits']
        summ['top_minus_bg'] = summ['top_total_hits'] - summ['bg_total_hits']
        summ.to_csv(out_dir / 'fimo_focus_family_summary.tsv', sep='\t', index=False)
        summary_rows.append({'metric':'fimo_top_n', 'value': int(args.top_n)})
        summary_rows.append({'metric':'fimo_ran', 'value': 1})
    else:
        summary_rows.append({'metric':'fimo_ran', 'value': 0})

    pd.DataFrame(summary_rows).to_csv(out_dir / 'hepg2_mech_validation_summary.tsv', sep='\t', index=False)
    print(f'[done] wrote mechanistic validation outputs -> {out_dir}')


if __name__ == '__main__':
    main()

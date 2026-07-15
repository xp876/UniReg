#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith('.gz'):
        return pd.read_csv(path, sep=None, engine='python', compression='gzip')
    return pd.read_csv(path, sep=None, engine='python')


def first_existing(df, names):
    for n in names:
        if n and n in df.columns:
            return n
    return None


def normalize_bool(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {'1','true','t','yes','y','emvar','sig','significant'}:
        return 1
    if s in {'0','false','f','no','n','non-emvar','nonsig','not_significant'}:
        return 0
    try:
        return int(float(s) > 0)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_tsv', required=True)
    ap.add_argument('--out_tsv', required=True)
    ap.add_argument('--cell_col', default='')
    ap.add_argument('--cell_value', default='HepG2')
    ap.add_argument('--id_col', default='')

    # wide mode
    ap.add_argument('--ref_activity_col', default='')
    ap.add_argument('--alt_activity_col', default='')
    ap.add_argument('--ref_seq_col', default='')
    ap.add_argument('--alt_seq_col', default='')

    # long mode
    ap.add_argument('--long_allele_col', default='')
    ap.add_argument('--long_activity_col', default='')
    ap.add_argument('--ref_allele_values', default='ref,REF,0,Ref')
    ap.add_argument('--alt_allele_values', default='alt,ALT,1,Alt')

    # optional merge-in sequence table
    ap.add_argument('--seq_table', default='')
    ap.add_argument('--seq_id_col', default='')

    # extra metadata
    ap.add_argument('--fdr_col', default='')
    ap.add_argument('--emvar_col', default='')
    args = ap.parse_args()

    df = read_table(args.in_tsv)

    if args.cell_col:
        df = df[df[args.cell_col].astype(str).str.lower() == str(args.cell_value).lower()].copy()

    id_col = args.id_col or first_existing(df, ['variant_id','var_id','rsid','id','variant','oligo_id'])
    if not id_col:
        raise SystemExit('Could not infer id column. Pass --id_col explicitly.')

    fdr_col = args.fdr_col or first_existing(df, ['fdr','padj','qval','q_value','adj_p'])
    emvar_col = args.emvar_col or first_existing(df, ['emvar','is_emvar','significant','is_significant'])

    if args.ref_activity_col and args.alt_activity_col:
        wide = df.copy()
        ref_activity_col = args.ref_activity_col
        alt_activity_col = args.alt_activity_col
        ref_seq_col = args.ref_seq_col or first_existing(wide, ['ref_seq','sequence_ref','ref_sequence','oligo_ref','sequence'])
        alt_seq_col = args.alt_seq_col or first_existing(wide, ['alt_seq','sequence_alt','alt_sequence','oligo_alt'])
    elif args.long_allele_col and args.long_activity_col:
        ref_vals = {x.strip().lower() for x in args.ref_allele_values.split(',') if x.strip()}
        alt_vals = {x.strip().lower() for x in args.alt_allele_values.split(',') if x.strip()}
        tmp = df.copy()
        tmp['_allele_norm'] = tmp[args.long_allele_col].astype(str).str.strip().str.lower()
        tmp = tmp[tmp['_allele_norm'].isin(ref_vals | alt_vals)].copy()
        tmp['_allele_role'] = np.where(tmp['_allele_norm'].isin(ref_vals), 'ref', 'alt')
        keep = [id_col, args.long_activity_col, '_allele_role']
        for c in [fdr_col, emvar_col, args.cell_col, args.ref_seq_col, args.alt_seq_col]:
            if c and c in tmp.columns and c not in keep:
                keep.append(c)
        sub = tmp[keep].copy()
        piv = sub.pivot_table(index=id_col, columns='_allele_role', values=args.long_activity_col, aggfunc='first').reset_index()
        piv.columns = [id_col, 'ref_activity', 'alt_activity']
        wide = piv.copy()
        ref_activity_col = 'ref_activity'
        alt_activity_col = 'alt_activity'
        if fdr_col and fdr_col in sub.columns:
            wide = wide.merge(sub[[id_col, fdr_col]].drop_duplicates(), on=id_col, how='left')
        if emvar_col and emvar_col in sub.columns:
            wide = wide.merge(sub[[id_col, emvar_col]].drop_duplicates(), on=id_col, how='left')
        ref_seq_col = ''
        alt_seq_col = ''
    else:
        raise SystemExit('Provide either wide-mode columns (--ref_activity_col/--alt_activity_col) or long-mode columns (--long_allele_col/--long_activity_col).')

    if args.seq_table:
        seq_df = read_table(args.seq_table)
        seq_id_col = args.seq_id_col or first_existing(seq_df, [id_col, 'variant_id','var_id','rsid','id'])
        if not seq_id_col:
            raise SystemExit('Could not infer sequence-table id column. Pass --seq_id_col explicitly.')
        sref = args.ref_seq_col or first_existing(seq_df, ['ref_seq','sequence_ref','ref_sequence','oligo_ref','ref'])
        salt = args.alt_seq_col or first_existing(seq_df, ['alt_seq','sequence_alt','alt_sequence','oligo_alt','alt'])
        if not (sref and salt):
            raise SystemExit('Could not infer ref/alt sequence columns from sequence table.')
        seq_sub = seq_df[[seq_id_col, sref, salt]].drop_duplicates().rename(columns={seq_id_col: id_col, sref: 'sequence_ref', salt: 'sequence_alt'})
        wide = wide.merge(seq_sub, on=id_col, how='left')
        ref_seq_col = 'sequence_ref'
        alt_seq_col = 'sequence_alt'

    out = pd.DataFrame()
    out['variant_id'] = wide[id_col].astype(str)
    out['element_id'] = 'R:' + out['variant_id']
    out['log2_WT'] = pd.to_numeric(wide[ref_activity_col], errors='coerce')
    out['log2_MT'] = pd.to_numeric(wide[alt_activity_col], errors='coerce')
    out['delta'] = out['log2_WT'] - out['log2_MT']
    out['mean'] = (out['log2_WT'] + out['log2_MT']) / 2.0
    out['fdr'] = pd.to_numeric(wide[fdr_col], errors='coerce') if fdr_col and fdr_col in wide.columns else np.nan
    out['emvar'] = wide[emvar_col].map(normalize_bool) if emvar_col and emvar_col in wide.columns else np.where(out['fdr'].notna() & (out['fdr'] < 0.1), 1, np.nan)
    if ref_seq_col and ref_seq_col in wide.columns:
        out['sequence_ref'] = wide[ref_seq_col].astype(str)
        out['sequence'] = out['sequence_ref']
    else:
        out['sequence_ref'] = np.nan
        out['sequence'] = np.nan
    if alt_seq_col and alt_seq_col in wide.columns:
        out['sequence_alt'] = wide[alt_seq_col].astype(str)
    else:
        out['sequence_alt'] = np.nan
    out['context'] = 'INT'
    out['sample_weight'] = 1.0

    out = out.dropna(subset=['variant_id','log2_WT','log2_MT','delta']).copy()
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_tsv, sep='\t', index=False)
    print(f'[done] standardized rows={len(out)} -> {args.out_tsv}')


if __name__ == '__main__':
    main()

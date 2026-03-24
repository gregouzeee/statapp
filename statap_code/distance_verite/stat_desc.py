#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_qa_numeric.py

Analyse descriptive spécialisée pour jeux de questions->réponses numériques.

Usage:
    python analyze_qa_numeric.py --in data.jsonl --format jsonl --outdir out_analysis

Entrées supportées:
 - JSONL (une ligne JSON contenant {"question": ..., "answer": ...})
 - CSV (colonnes 'question' et 'answer')

Sorties:
 - summaries.json
 - answers_summary.csv
 - question_stats.csv
 - outliers.csv
 - plots/* (histogrammes, barres par décennie, scatter length vs answer)
"""

import argparse
import json
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'figure.max_open_warning': 0})

# --- Helpers ---
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def is_integer_like(x):
    try:
        return float(x).is_integer()
    except Exception:
        return False

def detect_interrogative(q):
    q_low = q.lower()
    # simple set of WH words/fragments (english)
    whs = ['when', 'what year', 'in what year', 'how old', 'how many', 'how long', 'how far', 'where', 'who', 'which', 'what']
    found = [w for w in whs if w in q_low]
    if 'how old' in q_low or 'how many years' in q_low:
        return 'age'
    if 'what year' in q_low or 'in what year' in q_low or q_low.strip().startswith('when'):
        return 'year'
    if 'how many' in q_low or 'how long' in q_low:
        return 'quantity'
    if 'square' in q_low or 'area' in q_low or 'square miles' in q_low:
        return 'area'
    return found[0] if found else 'other'

def decade_from_year(y):
    try:
        y = int(y)
        return (y // 10) * 10
    except Exception:
        return None

def century_from_year(y):
    try:
        y = int(y)
        return (y // 100) + 1
    except Exception:
        return None

# --- Main analysis function ---
def analyze(df, outdir):
    outdir = ensure_dir(outdir)
    plots = ensure_dir(outdir/'plots')

    # Ensure answer numeric
    df = df.copy()
    df['answer_numeric'] = pd.to_numeric(df['answer'], errors='coerce')
    n_total = len(df)
    n_numeric = df['answer_numeric'].notna().sum()

    # Basic numeric summaries
    num = df['answer_numeric'].dropna()
    summary = {
        'n_total': int(n_total),
        'n_numeric': int(n_numeric),
        'fraction_numeric': float(n_numeric / max(1, n_total))
    }
    if len(num)>0:
        summary.update({
            'mean': float(num.mean()),
            'median': float(num.median()),
            'std': float(num.std()),
            'min': float(num.min()),
            'max': float(num.max()),
            'skew': float(num.skew()),
            'kurtosis': float(num.kurtosis()),
            'q25': float(num.quantile(0.25)),
            'q75': float(num.quantile(0.75)),
            'iqr': float(num.quantile(0.75) - num.quantile(0.25))
        })

    # integer-like / .0 detection
    df['is_integer_like'] = df['answer_numeric'].apply(lambda x: bool((not np.isnan(x)) and float(x).is_integer()) if not pd.isna(x) else False)
    summary['n_integer_like'] = int(df['is_integer_like'].sum())
    summary['fraction_integer_like'] = float(df['is_integer_like'].sum() / max(1, n_numeric))

    # decimals count
    def decimals_count(x):
        if pd.isna(x): return None
        s = str(x)
        if '.' in s:
            return len(s.split('.')[-1].rstrip('0'))  # ignore trailing zeros
        return 0
    df['decimals'] = df['answer'].apply(lambda a: None if pd.isna(a) else (0 if is_integer_like(a) else (len(str(a).split('.')[-1].rstrip('0')) if '.' in str(a) else 0)))

    # detect years, ages, large numbers heuristics
    def classify_numeric(x):
        if pd.isna(x):
            return 'missing'
        try:
            v = float(x)
        except Exception:
            return 'non-num'
        # heuristics
        if 1000 <= v <= 2100 and float(v).is_integer():
            return 'year'
        if 0 <= v <= 120 and float(v).is_integer():
            return 'age'
        if abs(v) >= 10000:
            return 'large_number'
        if 121 <= v <= 999:
            return 'medium_int'
        if v < 0:
            return 'negative'
        return 'other_numeric'

    df['answer_type'] = df['answer_numeric'].apply(classify_numeric)

    # decades / centuries where year-like
    df['decade'] = df['answer_numeric'].apply(lambda x: decade_from_year(x) if not pd.isna(x) and 1000<=x<=2100 else None)
    df['century'] = df['answer_numeric'].apply(lambda x: century_from_year(x) if not pd.isna(x) and 1000<=x<=2100 else None)

    # question characteristics
    df['q_len_chars'] = df['question'].fillna('').apply(len)
    df['q_len_words'] = df['question'].fillna('').apply(lambda s: len(s.split()))
    df['q_wh_type'] = df['question'].fillna('').apply(detect_interrogative)

    # outliers: IQR & z-score
    if len(num)>0:
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        df['outlier_iqr'] = df['answer_numeric'].apply(lambda v: False if pd.isna(v) else (v < lower_iqr or v > upper_iqr))
        df['zscore'] = df['answer_numeric'].apply(lambda v: float((v - num.mean()) / num.std()) if not pd.isna(v) and num.std() != 0 else None)
        df['outlier_z'] = df['zscore'].apply(lambda z: False if z is None else (abs(z) > 3))
        summary.update({
            'iqr_lower': float(lower_iqr),
            'iqr_upper': float(upper_iqr),
            'n_outliers_iqr': int(df['outlier_iqr'].sum()),
            'n_outliers_z': int(df['outlier_z'].sum())
        })

    # group counts and distributions
    type_counts = df['answer_type'].value_counts(dropna=False).to_dict()
    decade_counts = df['decade'].value_counts(dropna=False).sort_index().to_dict()

    # correlations
    corr_len_answer = None
    if df['answer_numeric'].notna().sum() > 1:
        try:
            corr_len_answer = float(df[['q_len_words','answer_numeric']].dropna().corr().iloc[0,1])
        except Exception:
            corr_len_answer = None

    # Save tables
    (outdir/'summaries.json').write_text(json.dumps(summary, indent=2))
    df.to_csv(outdir/'full_rows_with_features.csv', index=False)
    pd.DataFrame.from_dict(type_counts, orient='index', columns=['count']).to_csv(outdir/'answer_type_counts.csv')
    pd.DataFrame.from_dict(decade_counts, orient='index', columns=['count']).to_csv(outdir/'decade_counts.csv')

    # Save small aggregated tables
    question_stats = df[['question','q_len_chars','q_len_words','q_wh_type']].copy()
    question_stats.to_csv(outdir/'question_stats.csv', index=False)

    # Save outliers
    if 'outlier_iqr' in df.columns:
        df[df['outlier_iqr']].to_csv(outdir/'outliers_iqr.csv', index=False)
    if 'outlier_z' in df.columns:
        df[df['outlier_z']].to_csv(outdir/'outliers_z.csv', index=False)

    # Plots
    try:
        # histogram of answers
        plt.figure(figsize=(7,4))
        df['answer_numeric'].dropna().hist(bins=30)
        plt.title('Distribution of numeric answers')
        plt.xlabel('answer')
        plt.tight_layout()
        plt.savefig(plots/'hist_answers.png', dpi=150)
        plt.close()
    except Exception:
        pass

    try:
        # bar plot decades
        dec = df['decade'].dropna().astype(int).value_counts().sort_index()
        if len(dec)>0:
            plt.figure(figsize=(8,3))
            dec.plot(kind='bar')
            plt.title('Counts by decade (for year-like answers)')
            plt.xlabel('decade')
            plt.tight_layout()
            plt.savefig(plots/'bar_decades.png', dpi=150)
            plt.close()
    except Exception:
        pass

    try:
        # scatter question length vs answer
        plt.figure(figsize=(6,4))
        plt.scatter(df['q_len_words'], df['answer_numeric'], s=12, alpha=0.6)
        plt.xlabel('question length (words)')
        plt.ylabel('answer numeric')
        plt.title('Question length vs numeric answer')
        plt.tight_layout()
        plt.savefig(plots/'scatter_q_len_vs_answer.png', dpi=150)
        plt.close()
    except Exception:
        pass

    # quick textual report
    report_lines = []
    report_lines.append(f"Rows total: {summary['n_total']}")
    report_lines.append(f"Numeric answers: {summary['n_numeric']} ({summary['fraction_numeric']*100:.1f}%)")
    if 'mean' in summary:
        report_lines.append(f"Answer mean={summary['mean']:.2f}, median={summary['median']:.2f}, std={summary['std']:.2f}")
    report_lines.append(f"Integer-like answers (.0): {summary['n_integer_like']} ({summary['fraction_integer_like']*100:.1f}%)")
    report_lines.append(f"Detected types: {type_counts}")
    if corr_len_answer is not None:
        report_lines.append(f"Correlation question length (words) vs answer: {corr_len_answer:.3f}")

    (outdir/'report_summary.txt').write_text("\n".join(report_lines))

    print("Analysis finished. Outputs in:", outdir)

# --- CLI ---
def load_input(path, fmt):
    path = Path(path)
    if fmt == 'jsonl':
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    # try eval fallback
                    records.append(eval(line))
        return pd.DataFrame(records)
    elif fmt == 'csv':
        return pd.read_csv(path)
    else:
        raise ValueError("format must be 'jsonl' or 'csv'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze QA numeric dataset")
    parser.add_argument('--in', dest='infile', required=True, help="input file (jsonl or csv)")
    parser.add_argument('--format', dest='fmt', choices=['jsonl','csv'], default='jsonl', help='input format (jsonl or csv)')
    parser.add_argument('--outdir', dest='outdir', default='analysis_out', help='output folder')
    args = parser.parse_args()
    df = load_input(args.infile, args.fmt)
    # Expect columns question and answer
    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("input must contain 'question' and 'answer' columns/keys")
    analyze(df[['question','answer']], args.outdir)
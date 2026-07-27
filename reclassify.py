#!/usr/bin/env python3
"""Re-derive the attention/dense/other split from saved kernels_*.json.

No GPU / no model runs — reads the persisted per-kernel names+times and
recomputes summary.json using the current classify() rules. Use --inspect
to dump the largest kernels currently landing in 'other' (to tune rules).
"""
import json, glob, sys, os
from collections import defaultdict
from profile_gap import classify, OUT


def recompute(kernels):
    for r in kernels:
        r['op_class'] = classify(r['kernel_name'].lower())
    total = sum(r['cuda_time_us'] for r in kernels if r['op_class'] != 'noncompute')
    g = lambda c: sum(r['cuda_time_us'] for r in kernels if r['op_class'] == c)
    attn, dense, other, noncompute = g('attention'), g('dense'), g('other'), g('noncompute')
    return {
        'total_time_ms': total / 1000,
        'attention_time_ms': attn / 1000,
        'dense_time_ms': dense / 1000,
        'other_time_ms': other / 1000,
        'noncompute_time_ms': noncompute / 1000,
        'attention_pct': 100 * attn / total if total > 0 else 0,
        'dense_pct': 100 * dense / total if total > 0 else 0,
        'other_pct': 100 * other / total if total > 0 else 0,
    }


def main():
    inspect = '--inspect' in sys.argv
    files = sorted(glob.glob(f"{OUT}/kernels_*.json"))
    if inspect:
        agg = defaultdict(float)
        for f in files:
            for r in json.load(open(f))['kernels']:
                if classify(r['kernel_name'].lower()) == 'other':
                    agg[r['kernel_name']] += r['cuda_time_us']
        print("=== TOP 'other' KERNELS (aggregate ms across all models) ===")
        for name, us in sorted(agg.items(), key=lambda x: -x[1])[:25]:
            print(f"  {us/1000:8.1f}ms | {name[:90]}")
        return

    # Rewrite summary.json rows from saved kernels (models present on disk).
    summary_path = f"{OUT}/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else []
    by_key = {}
    for f in files:
        d = json.load(open(f))
        by_key[(d['model'], d['batch'])] = recompute(d['kernels'])
    updated = 0
    for row in summary:
        k = (row.get('model'), row.get('batch'))
        if row.get('status') == 'success' and k in by_key:
            row.update(by_key[k]); updated += 1
            r = by_key[k]
            print(f"  {row['model']:18s} attn={r['attention_pct']:5.1f}% "
                  f"dense={r['dense_pct']:5.1f}% other={r['other_pct']:5.1f}%")
    json.dump(summary, open(summary_path, 'w'), indent=2)
    print(f"Rewrote {updated} rows in summary.json")


if __name__ == "__main__":
    main()

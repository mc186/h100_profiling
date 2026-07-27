import json, glob
for f in sorted(glob.glob("/workspace/results/kernels_*_b1.json")):
    d = json.load(open(f))
    ks = sorted(d["kernels"], key=lambda r: -r["cuda_time_us"])[:10]
    tot = sum(r["cuda_time_us"] for r in d["kernels"])
    print("====", d["model"], "total_ms", round(tot/1000,1))
    for r in ks:
        print(f"  {r['cuda_time_us']/1000:7.1f}ms  {r['op_class']:9s} | {r['kernel_name'][:75]}")

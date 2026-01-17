import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def plot_expert_util(rows, out_dir):
    srv = [r for r in rows if r.get("type") == "server_round"]
    if not srv:
        return
    data = []
    for r in srv:
        rnd = r["round"]
        util = r["util_frac"]
        for k, v in util.items():
            data.append({"round": rnd, "expert": int(k), "util": float(v)})
    df = pd.DataFrame(data)
    plt.figure()
    for k in sorted(df["expert"].unique()):
        sub = df[df["expert"] == k].sort_values("round")
        plt.plot(sub["round"], sub["util"], label=f"e{k}")
    plt.xlabel("round")
    plt.ylabel("utilization fraction")
    plt.title("Expert utilization over rounds")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "expert_utilization.png"), dpi=200)
    plt.close()

def plot_worst_client(rows, out_dir):
    ev = [r for r in rows if r.get("type") == "client_eval"]
    if not ev:
        return
    df = pd.DataFrame(ev)
    # PPL: lower better; worst is max
    grp = df.groupby("round")["ppl"].max().reset_index(name="worst_ppl")
    plt.figure()
    plt.plot(grp["round"], grp["worst_ppl"])
    plt.xlabel("round")
    plt.ylabel("worst client PPL (max across clients)")
    plt.title("Worst-client PPL over rounds")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "worst_client.png"), dpi=200)
    plt.close()

def plot_negative_transfer(rows, out_dir):
    ev = [r for r in rows if r.get("type") == "client_eval" and "neg_transfer_ppl" in r]
    if not ev:
        return
    df = pd.DataFrame(ev)
    last_round = df["round"].max()
    df = df[df["round"] == last_round].sort_values("client_id")
    plt.figure()
    plt.bar(df["client_id"], df["neg_transfer_ppl"])
    plt.axhline(0.0)
    plt.xlabel("client")
    plt.ylabel("neg transfer (PPL gain = local - fed)")
    plt.title("Negative transfer on PPL (final round)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "negative_transfer.png"), dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    args = ap.parse_args()

    path = os.path.join(args.log_dir, "metrics.jsonl")
    rows = read_jsonl(path)
    os.makedirs(args.log_dir, exist_ok=True)

    plot_expert_util(rows, args.log_dir)
    plot_worst_client(rows, args.log_dir)
    plot_negative_transfer(rows, args.log_dir)
    print("Saved plots to:", args.log_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import glob
import os
import re

EPOCH = 10

PATTERN = "uas_results/results_by_head_*_GPT2_*_53-ckpt-10-sas_preds.tsv"

rows = []

for path in glob.glob(PATTERN):

    fname = os.path.basename(path)
    m = re.match(r"results_by_head_(.+)_GPT2_(.+)_53-ckpt-10-sas_preds.tsv", fname)
    if not m:
        continue

    uid = m.group(1)
    condition = m.group(2)

    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            layer = int(row["layer"])
            head_cols = [k for k in row.keys() if k.startswith("h")]
            head_scores = {h: float(row[h]) for h in head_cols}

            max_head = max(head_scores, key=head_scores.get)
            max_score = head_scores[max_head]

            rows.append({
                "condition": condition,
                "uid": uid,
                "epoch": EPOCH,
                "layer": layer,
                "max_head": max_head,
                "max_score": max_score,
            })

out_file = "fw_attn_max_by_layer_epoch10.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "uid", "epoch", "layer", "max_head", "max_score"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_file}")
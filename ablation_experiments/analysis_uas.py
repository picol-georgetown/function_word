#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from collections import defaultdict

EPOCH = 10

conditions = [
    "natural_function",
    "no_function",
    "random_function",
    "five_function",
    "within_boundary",
    "more_function",
    "bigram_function",
]

PHENOMENON_GROUPS = {
    "anaphor_agreement": [
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
    ],
    "argument_structure": [
        "animate_subject_passive",
        "animate_subject_trans",
        "causative",
        "drop_argument",
        "inchoative",
        "intransitive",
        "passive_1",
        "passive_2",
        "transitive",
    ],
    "binding": [
        "principle_A_c_command",
        "principle_A_case_1",
        "principle_A_case_2",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
    ],
    "control_raising": [
        "existential_there_object_raising",
        "existential_there_subject_raising",
        "expletive_it_object_raising",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
    ],
    "determiner_noun_agreement": [
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adjective_1",
        "determiner_noun_agreement_with_adjective_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
    ],
    "ellipsis": [
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
    ],
    "filler_gap": [
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ],
    "irregular_forms": [
        "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs",
    ],
    "island_effects": [
        "adjunct_island",
        "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "left_branch_island_echo_question",
        "left_branch_island_simple_question",
        "sentential_subject_island",
        "wh_island",
    ],
    "npi_licensing": [
        "matrix_question_npi_licensor_present",
        "npi_present_1",
        "npi_present_2",
        "only_npi_licensor_present",
        "only_npi_scope",
        "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope",
    ],
    "quantifiers": [
        "existential_there_quantifiers_1",
        "existential_there_quantifiers_2",
        "superlative_quantifiers_1",
        "superlative_quantifiers_2",
    ],
    "subject_verb_agreement": [
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
}

UID2CAT = {}
for cat, uids in PHENOMENON_GROUPS.items():
    for u in uids:
        UID2CAT[u] = cat


def read_best_head_tsv(tsv_path):
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader)
        best_layer = int(row["best_layer"])
        best_head = int(row["best_head"])
        best_uas = float(row["best_uas"])
    return best_layer, best_head, best_uas


rows = []

INPUT_DIR = "."

for cond in conditions:
    for uid, cat in UID2CAT.items():
        if cat == "determiner_noun_agreement":
            continue

        fname = (
            f"uas_results/results_best_head_{uid}_GPT2_{cond}_53-ckpt-{EPOCH}-sas_preds.tsv"
        )
        tsv_path = os.path.join(INPUT_DIR, fname)

        if not os.path.exists(tsv_path):
            # print("Missing file:", tsv_path)
            continue

        best_layer, best_head, best_uas = read_best_head_tsv(tsv_path)

        rows.append(
            {
                "condition": cond,
                "category": cat,
                "uid": uid,
                "epoch": EPOCH,
                "best_layer": best_layer,
                "best_head": best_head,
                "best_uas": best_uas,
            }
        )


from collections import defaultdict

by_condition = defaultdict(list)
for r in rows:
    by_condition[r["condition"]].append(float(r["best_uas"]))

overall_rows = []
for cond, vals in by_condition.items():
    overall_rows.append(
        {
            "condition": cond,
            "category": "overall",
            "uid": "overall",
            "epoch": EPOCH,
            "best_layer": -1,
            "best_head": "overall",
            "best_uas": sum(vals) / len(vals),
        }
    )

rows_extended = rows + overall_rows

out_file = "blimp_uas_long_epoch10.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "category",
            "uid",
            "epoch",
            "best_layer",
            "best_head",
            "best_uas",
        ],
    )
    writer.writeheader()
    writer.writerows(rows_extended)

print(f"Wrote {len(rows_extended)} rows to {out_file}")

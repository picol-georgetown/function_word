import csv
import os
from collections import defaultdict
from statistics import mean, pstdev

SEEDS = [42, 53, 67]
EPOCH = 10

p_d = {
    "natural_function": "natural_function_blimp_results/",
    "no_function": "no_function_blimp_results/",
    "random_function": "random_function_blimp_results/",
    "five_function": "five_function_blimp_results/",
    "within_boundary": "within_boundary_blimp_results/",
    "more_function": "more_function_blimp_results/",
    "bigram_function": "bigram_function_blimp_results/",
}

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


def read_scores(csv_path):
    out = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for task, score in reader:
            uid = task.replace("blimp_", "")
            out[uid] = float(score)
    return out


rows = []
missing = []

for cond, folder in p_d.items():
    for seed in SEEDS:
        fname = f"results_GPT2_{cond}_{seed}_epoch-{EPOCH}.csv"
        csv_path = os.path.join(folder, fname)
        if not os.path.exists(csv_path):
            missing.append(csv_path)
            continue

        scores = read_scores(csv_path)

        for uid, acc in scores.items():
            cat = UID2CAT.get(uid, "unknown")
            if cat == "determiner_noun_agreement":
                continue
            if 'quantifier' in cat:
                continue

            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "category": cat,
                    "uid": uid,
                    "epoch": EPOCH,
                    "accuracy": acc,
                }
            )


by_cond_seed = defaultdict(list)  # (cond, seed) -> [acc...]
for r in rows:
    by_cond_seed[(r["condition"], r["seed"])].append(r["accuracy"])

for (cond, seed), vals in by_cond_seed.items():
    rows.append(
        {
            "condition": cond,
            "seed": seed,
            "category": "overall",
            "uid": "overall",
            "epoch": EPOCH,
            "accuracy": mean(vals),
        }
    )


bucket = defaultdict(list)
for r in rows:
    key = (r["condition"], r["category"], r["uid"], r["epoch"])
    bucket[key].append(r["accuracy"])

agg_rows = []
for (cond, cat, uid, epoch), vals in bucket.items():
    agg_rows.append(
        {
            "condition": cond,
            "category": cat,
            "uid": uid,
            "epoch": epoch,
            "n_seeds": len(vals),
            "mean_acc": mean(vals),
            "std_acc": pstdev(vals) if len(vals) > 1 else 0.0,
        }
    )

os.makedirs("overall_results", exist_ok=True)

long_file = f"overall_results/blimp_long_epoch{EPOCH}.csv"
with open(long_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "seed", "category", "uid", "epoch", "accuracy"],
    )
    writer.writeheader()
    writer.writerows(rows)

agg_file = f"overall_results/blimp_agg_epoch{EPOCH}.csv"
with open(agg_file, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "category", "uid", "epoch", "n_seeds", "mean_acc", "std_acc"],
    )
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"Wrote long table: {long_file}  (rows={len(rows)})")
print(f"Wrote agg table : {agg_file}   (rows={len(agg_rows)})")

if missing:
    print("\n[WARN] Missing files:")
    for p in missing:
        print("  -", p)
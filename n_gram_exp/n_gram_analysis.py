import csv
import os
from collections import defaultdict

p_d = {
    "natural_function": "natural_function_blimp_results/",
    "no_function": "no_function_blimp_results/",
    "random_function": "random_function_blimp_results/",
    "five_function": "five_function_blimp_results/",
    "within_boundary": "within_boundary_blimp_results/",
    "more_function": "more_function_blimp_results/",
    "bigram_function": "bigram_function_blimp_results/",
    "five_function_within_boundary": "five_function_within_boundary_blimp_results/",
    "five_function_random_function": "five_function_random_function_blimp_results/",
    "bigram_function_within_boundary": "bigram_function_within_boundary_blimp_results/",
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
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for task, score in reader:
            uid = task.replace("blimp_", "")
            out[uid] = float(score)
    return out


rows = []
missing = []

for cond, folder in p_d.items():
    csv_path = os.path.join(folder, f"results_n_gram/{cond}.binary_best.csv")
    if not os.path.exists(csv_path):
        missing.append(csv_path)
        continue

    scores = read_scores(csv_path)

    for uid, acc in scores.items():
        cat = UID2CAT.get(uid, "unknown")

        if cat == "determiner_noun_agreement":
            continue
        if cat == "quantifiers":
            continue

        rows.append(
            {
                "condition": cond,
                "category": cat,
                "uid": uid,
                "accuracy": round(acc*100,1),
            }
        )

uid_out = "n_gram_blimp_scores.csv"
with open(uid_out, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "category", "uid", "accuracy"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {uid_out}")

cat_scores = defaultdict(lambda: defaultdict(list))
for row in rows:
    cat_scores[row["condition"]][row["category"]].append(row["accuracy"])

categories = sorted({
    row["category"] for row in rows
})

combined_rows = []
for cond in p_d.keys():
    if cond not in cat_scores:
        continue

    row_out = {"condition": cond}
    cat_means = []

    for cat in categories:
        if cat in cat_scores[cond]:
            mean_score = round(sum(cat_scores[cond][cat]) / len(cat_scores[cond][cat]), 1)
            row_out[cat] = mean_score
            cat_means.append(mean_score)
        else:
            row_out[cat] = ""

    row_out["overall"] = round(sum(cat_means) / len(cat_means),1) if cat_means else ""
    combined_rows.append(row_out)

combined_out = "n_gram_blimp_category_overall_scores.csv"
with open(combined_out, "w", newline="") as f:
    fieldnames = ["condition"] + categories + ["overall"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(combined_rows)

print(f"Saved combined table to {combined_out}")

if missing:
    print("\nMissing files:")
    for p in missing:
        print(p)
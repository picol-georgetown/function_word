library(tidyverse)
library(patchwork) 

# token deletion ablation analysis
#/Users/xiulinyang/Desktop/TODO/function_word/overall_results/blimp_function_word_deletion_epoch10.csv
#/Users/xiulinyang/Desktop/TODO/function_word/overall_results/blimp_function_word_mask_epoch10.csv
df <- read_csv("/Users/xiulinyang/Desktop/TODO/function_word/blimp_function_word_deletion_epoch10.csv")

cat_seed <- df %>%
  group_by(seed, condition, category) %>%
  summarise(
    acc = mean(accuracy*100, na.rm = TRUE),
    .groups = "drop"
  )


delta_seed <- cat_seed %>%
  pivot_wider(names_from = condition, values_from = acc) %>%
  mutate(
    delta_natural  = ablation_natural          - baseline_natural,
    delta_more     = ablation_more    - baseline_more,
    delta_boundary = ablation_boundary  - baseline_boundary,
    delta_random   = ablation_random  - baseline_random,
    delta_bigram   = ablation_bigram  - baseline_bigram,
    delta_five     = ablation_five    - baseline_five
  ) %>%
  select(seed, category, starts_with("delta_")) %>%
  pivot_longer(
    cols = starts_with("delta_"),
    names_to = "condition",
    values_to = "delta"
  )

delta_mean <- delta_seed %>%
  group_by(category, condition) %>%
  summarise(
    delta = mean(delta, na.rm = TRUE),
    delta_sd = sd(delta, na.rm = TRUE),  # 可选
    .groups = "drop"
  )


# delta_mean <- delta_mean %>%
#   mutate(
#     label = ifelse(
#       category == "overall",
#       sprintf("%.1f\n(±%.1f)", delta, delta_sd),
#       sprintf("%.1f", delta)
#     )
#   )

delta_mean <- delta_mean %>%
  mutate(
    condition = recode(
      condition,
      "delta_natural"  = "Natural",
      "delta_more"     = "More Function",
      "delta_boundary" = "Within Boundary",
      "delta_random"   = "Random Function",
      "delta_bigram"   = "Bigram Function",
      "delta_five"     = "Five Function"
    ),
    condition = factor(
      condition,
      levels = c(
        "Natural",
        "Five Function",
        "More Function",
        "Bigram Function",
        "Random Function",
        "Within Boundary"
      )
    ),
    category = fct_recode(
      category,
      "Overall"           = "overall",
      "S-V Agreement"     = "subject_verb_agreement",
      "Irregular Forms"   = "irregular_forms",
      "NPI Licensing"     = "npi_licensing",
      "Island Effects"    = "island_effects",
      "Filler–Gap"        = "filler_gap",
      "Ellipsis"          = "ellipsis",
      "Control & Raising" = "control_raising",
      "Binding"           = "binding",
      "Arg. Structure"    = "argument_structure",
      "Ana. Agreement"    = "anaphor_agreement"
    ),
    category = factor(
      category,
      levels = c(
        "Overall",
        "S-V Agreement",
        "Irregular Forms",
        "NPI Licensing",
        "Island Effects",
        "Filler–Gap",
        "Ellipsis",
        "Control & Raising",
        "Binding",
        "Arg. Structure",
        "Ana. Agreement"
      )
    )
  )






p <- ggplot(delta_mean, aes(x = condition, y = category, fill = delta)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f", delta)), size = 4) +
  scale_fill_gradient2(
    low = "#B64C4C",
    mid = "white",
    high = "#6D8CD1",
    midpoint = 0,
    name = expression(Delta * "Accuracy")
  ) +
  labs(
    x = "Condition",
    y = "Phenomenon category"
  ) +
  theme_minimal(base_size = 15) +
  theme(
    # axis.text.x  = element_blank(),
    # axis.ticks.x = element_blank(),
    # axis.title.x = element_blank(),
    axis.text.x  = element_text(angle = 30, hjust = 1, face = "bold", size = 15),
    axis.text.y  = element_text(face = "bold", size = 15),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    panel.grid   = element_blank()
  )

p


library(tidyverse)
dat_global <- df %>%
  filter(category != "overall") %>%
  group_by(seed, condition, category) %>%
  summarise(cat_acc = mean(accuracy * 100, na.rm = TRUE), .groups = "drop") %>%
  group_by(seed, condition) %>%
  summarise(acc = mean(cat_acc, na.rm = TRUE), .groups = "drop")
conditions <- c("natural", "five", "more", "bigram", "random", "boundary")
global_stats <- tibble()

for (cond in conditions) {
  baseline_name <- paste0("baseline_", cond)
  ablation_name <- paste0("ablation_", cond)
  
  b_scores <- dat_global %>% filter(condition == baseline_name) %>% arrange(seed) %>% pull(acc)
  a_scores <- dat_global %>% filter(condition == ablation_name) %>% arrange(seed) %>% pull(acc)
  
  t_res <- t.test(a_scores, b_scores, paired = TRUE)
  
  global_stats <- bind_rows(global_stats, tibble(
    condition = cond,
    mean_baseline = mean(b_scores),
    mean_ablation = mean(a_scores),
    delta = mean(a_scores - b_scores),
    p_val = t_res$p.value,
    t_stat = t_res$statistic
  ))
}

print(global_stats)

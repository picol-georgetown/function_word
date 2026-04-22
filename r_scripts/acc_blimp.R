library(tidyverse)
library(patchwork)

df <- read_csv("/Users/xiulinyang/Desktop/TODO/function_word/overall_results/blimp_long_epoch10.csv")

cat_seed <- df %>%
  group_by(seed, condition, category) %>%
  summarise(acc = mean(accuracy, na.rm = TRUE), .groups = "drop")

cat_means <- cat_seed %>%
  group_by(condition, category) %>%
  summarise(
    acc = mean(acc, na.rm = TRUE),
    acc_sd = sd(acc, na.rm = TRUE),
    .groups = "drop"
    
  )

delta_df <- cat_means %>%
  select(category, condition, acc) %>%
  pivot_wider(names_from = condition, values_from = acc) %>%
  mutate(
    delta_no       = no_function        - natural_function,
    delta_random   = random_function    - natural_function,
    delta_five     = five_function      - natural_function,
    delta_boundary = within_boundary    - natural_function,
    delta_more     = more_function      - natural_function,
    delta_bigram   = bigram_function    - natural_function
  ) %>%
  select(category, delta_no, delta_random, delta_five,
         delta_boundary, delta_more, delta_bigram) %>%
  pivot_longer(
    cols = starts_with("delta_"),
    names_to = "condition",
    values_to = "delta"
  ) %>%
  mutate(
    delta = delta * 100,
    condition = recode(
      condition,
      "delta_no"       = "No Function",
      "delta_random"   = "Random Function",
      "delta_five"     = "Five Function",
      "delta_boundary" = "Within Boundary",
      "delta_more"     = "More Function",
      "delta_bigram"   = "Bigram Function"
    ),
    condition = factor(
      condition,
      levels = c(
        "No Function",
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

p_acc <- ggplot(delta_df, aes(x = condition, y = category, fill = delta)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f", delta)), size = 3) +
  scale_fill_gradient2(
    low = "#B64C4C",
    mid = "white",
    high = "#6D8CD1",
    midpoint = 0,
    name = expression(Delta * "Accuracy")
  ) +
  labs(x = "Condition", y = "Phenomenon Category", title = "") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x  = element_text(angle = 30, hjust = 1, face = "bold", size = 11),
    axis.text.y  = element_text(face = "bold", size = 11),
    axis.title.x = element_text(face = "bold", size = 13),
    axis.title.y = element_text(face = "bold", size = 13),
    panel.grid   = element_blank()
  )

p_acc

paired_df <- cat_seed %>%
  select(seed, condition, category, acc) %>%
  pivot_wider(names_from = condition, values_from = acc)

library(broom)

tests <- paired_df %>%
  pivot_longer(
    cols = -c(seed, category, natural_function),
    names_to = "condition",
    values_to = "acc_other"
  ) %>%
  group_by(category, condition) %>%
  summarise(
    t = list(t.test(acc_other, natural_function, paired = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(res = map(t, broom::tidy)) %>%
  unnest(res)

tests <- tests %>%
  group_by(category) %>%
  mutate(p.adj = p.adjust(p.value, method = "holm")) %>%
  ungroup()
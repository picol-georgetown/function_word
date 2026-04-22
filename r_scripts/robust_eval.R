library(tidyverse)
library(patchwork) 
df <- read_csv("blimp_natural_ablation_epoch10_53.csv")

cat_means <- df %>%
  group_by(category, condition) %>%
  summarise(acc = mean(accuracy), .groups = "drop")

delta_df <- cat_means %>%
  pivot_wider(names_from = condition, values_from = acc) %>%
  mutate(
    delta_1 = baseline_no_function - baseline_natural,
    delta_2 = baseline_random_function -baseline_natural,
    delta_3 = baseline_within_boundary - baseline_natural,
    delta_4 = baseline_bigram_function -baseline_natural,
    delta_5 = baseline_five_function - baseline_natural
    
  ) %>%
  select(category, delta_1, delta_2, delta_3, delta_4,delta_5) %>%
  pivot_longer(
    cols      = starts_with("delta_"),
    names_to  = "condition",
    values_to = "delta"
  ) %>%
  mutate(
    condition = recode(
      condition,
      "delta_1" = "No Function",
      "delta_2" = "Random Function",
      "delta_3" = "Within Boundary",
      "delta_4" = "Bigram Function",
      "delta_5" = "Five Function",
    ),
    condition = factor(
      condition,
      levels = c("No Function", "Random Function", "Within Boundary", "Bigram Function", "Five Function")
    ),
    category = fct_recode(
      category,
      "Overall"                   = "overall",
      "Subject–Verb Agreement"    = "subject_verb_agreement",
      "Irregular Forms"           = "irregular_forms",
      "Quantifiers"               = "quantifiers",
      "NPI Licensing"             = "npi_licensing",
      "Island Effects"            = "island_effects",
      "Filler–Gap"                = "filler_gap",
      "Ellipsis"                  = "ellipsis",
      "Control & Raising"         = "control_raising",
      "Binding"                   = "binding",
      "Argument Structure"        = "argument_structure",
      "Anaphor Agreement"         = "anaphor_agreement"
    ),
    category = factor(
      category,
      levels = c(
        "Overall",
        "Subject–Verb Agreement",
        "Irregular Forms",
        "Quantifiers",
        "NPI Licensing",
        "Island Effects",
        "Filler–Gap",
        "Ellipsis",
        "Control & Raising",
        "Binding",
        "Argument Structure",
        "Anaphor Agreement"
      )
    )
  )


p <- ggplot(delta_df, aes(x = condition, y = category, fill = delta)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.3f", delta)), size = 3) +
  scale_fill_gradient2(
    low = "#B64C4C",
    mid = "white",
    high = "#6D8CD1",
    midpoint  = 0,
    name = expression(Delta * "Accuracy")
  ) +
  labs(
    x     = "Tests set",
    y     = "Phenomenon category",
    title = ""
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(size = 10, face = "bold"),
    axis.text.x  = element_text(angle = 30, hjust = 1, face = "bold", size = 11),
    axis.text.y  = element_text(face = "bold", size = 11),
    axis.title.x = element_text(face = "bold", size = 13),
    axis.title.y = element_text(face = "bold", size = 13),
    panel.grid   = element_blank()
  )


p


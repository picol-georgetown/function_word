library(tidyverse)

df <- read_csv("/Users/xiulinyang/Desktop/TODO/function_word/function_head_53.csv") %>%
  filter(freq > 0) %>%
  mutate(
    layer = as.numeric(layer),
    head  = as.numeric(head)
  ) %>%
  mutate(
    condition = factor(
      condition,
      levels = c(
        "natural_function",
        "more_function",
        "five_function",
        "random_function",
        "bigram_function",
        "within_boundary"
      ),
      labels = c(
        "Natural Function",
        "More Function",
        "Five Function",
        "Random Function",
        "Bigram Function",
        "Within-Boundary"
      )
    )
  )

cond_colors <- c(
  "Natural Function"  = "#4D4D4D",
  "More Function"     = "#E6AB02",
  "Five Function"     = "#EACC83",
  "Random Function"   = "#B4D388",
  "Bigram Function"   = "#43B58A",
  "Within-Boundary"   = "#0082B9"
)

df_plot <- df %>%
  mutate(freq_size = log1p(freq))

df_top <- df_plot %>%
  group_by(condition) %>%
  slice_max(order_by = freq, n = 5, with_ties = FALSE) %>%
  ungroup()

p <- ggplot(df_plot, aes(x = layer, y = head)) +
  geom_point(
    aes(size = freq_size, color = condition),
    alpha = 0.65
  ) +
  geom_text(
    data = df_top,
    aes(label = freq),
    color = "black",
    fontface = "bold",
    size = 3,
    vjust = 0.5,
    hjust = 0.5,
    show.legend = FALSE
  ) +
  scale_size(range = c(0.6, 12)) +
  scale_color_manual(values = cond_colors) +
  scale_x_continuous(breaks = 0:11) +
  scale_y_reverse(breaks = 0:11) +
  coord_fixed() +
  facet_wrap(~ condition, nrow = 1) +
  labs(x = "Layer", y = "Head") +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(size = 7),
    axis.text.y = element_text(size = 7),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold", size = 10),
    legend.position = "none"
  )

p
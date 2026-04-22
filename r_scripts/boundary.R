library(tidyverse)
library(ggrepel)

df_closed <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/boundary_stats_function.tsv",
  show_col_types = FALSE
) %>%
  mutate(
    Boundary_Rate = as.numeric(Boundary_Rate),
    Type = "Function"
  ) %>%
  drop_na(Boundary_Rate)

df_open <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/boundary_stats_open.tsv",
  show_col_types = FALSE
) %>%
  mutate(
    Boundary_Rate = as.numeric(Boundary_Rate),
    Type = "Content"
  ) %>%
  drop_na(Boundary_Rate)

df_all <- bind_rows(df_closed, df_open)


# --- outliers ---
low_closed <- df_closed %>%
  filter(Boundary_Rate < 0.0)

high_open <- df_open %>%
  filter(Boundary_Rate > 1.0)

p <- ggplot(df_all, aes(x = Type, y = Boundary_Rate, fill = Type)) +

  ## violin
  geom_violin(
    alpha = 0.35,
    width = 0.9,
    trim = FALSE
  ) +

  ## jitter
  geom_jitter(
    width = 0.12,
    alpha = 0.25,
    size = 1.2,
    color = "gray35"
  ) +

  ## Closed: low outliers
  geom_point(
    data = low_closed,
    aes(x = "Function", y = Boundary_Rate),
    color = "#B64C4C",
    size = 2.2,
    alpha = 0.9,
    inherit.aes = FALSE
  ) +

  geom_text_repel(
    data = low_closed,
    aes(x = "Function", y = Boundary_Rate, label = Language),
    size = 3,
    min.segment.length = 0,
    box.padding = 0.35,
    point.padding = 0.25,
    max.overlaps = Inf,
    inherit.aes = FALSE
  ) +

  ## Open: high outliers
  geom_point(
    data = high_open,
    aes(x = "Content", y = Boundary_Rate),
    color = "#B64C4C",
    size = 2.2,
    alpha = 0.9,
    inherit.aes = FALSE
  ) +

  geom_text_repel(
    data = high_open,
    aes(x = "Content", y = Boundary_Rate, label = Language),
    size = 3,
    min.segment.length = 0,
    box.padding = 0.35,
    point.padding = 0.25,
    max.overlaps = Inf,
    inherit.aes = FALSE
  ) +

  ## reference lines
  # geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50") +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  coord_flip() +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2),
    labels = scales::number_format(accuracy = 0.1)
  )+
  scale_fill_manual(
    values = c(
      "Function" = "#6D8CD1",
      "Content"   = "#C74A4A"
    )
  ) +

  labs(
    x = NULL,
    y = "Boundary Rate",
    fill = NULL
  ) +

  theme_minimal(base_size = 15) +
  theme(
    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.8
    ),
    panel.grid.minor = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "none",
    axis.text.y = element_text(
      angle = 90,
      size  = 15
    )
  )

median_closed <- median(df_closed$Boundary_Rate, na.rm = TRUE)
median_open   <- median(df_open$Boundary_Rate, na.rm = TRUE)

median_closed
median_open

p
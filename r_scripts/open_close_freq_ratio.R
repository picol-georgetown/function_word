library(tidyverse)

df_fw <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/close_vs_open_count.tsv",
  show_col_types = FALSE
)

df_long <- df_fw %>%
  transmute(
    language,
    closed_type  = closed_ratio,
    closed_freq  = closed_freq_ratio,
    open_type    = open_ratio,
    open_freq    = open_freq_ratio
  ) %>%
  pivot_longer(
    cols = -language,
    names_to = c("class", ".value"),
    names_pattern = "(closed|open)_(type|freq)"
  ) %>%
  mutate(
    class = recode(
      class,
      "closed" = "Function words",
      "open"   = "Content words"
    )
  )

ggplot(df_long, aes(
  x = type,
  y = freq,
  color = class
)) +
  geom_point(
    size = 2,
    alpha = 0.75
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 0.7
  ) +
  scale_color_manual(
    values = c(
      "Function words" = "#6D8CD1",
      "Content words"    = "#B64C4C"
    )
  ) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_fixed() +
  labs(
    x = "Type Ratio",
    y = "Token Frequency Ratio",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    legend.position = c(0.97, 0.03),
    legend.justification = c("right", "bottom"),
    # legend.background = element_rect(
    #   fill = "white",
    #   color = "black",
    #   linewidth = 0.4
    # ),
    legend.key = element_blank(),
    legend.text = element_text(size = 15),

    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18)
  )


library(tidyverse)
library(ggrepel)

df_entropy <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/dependency_entropy_stats.tsv",
  show_col_types = FALSE
)

lims <- range(
  c(df_entropy$Dep_Closed_Entropy, df_entropy$Dep_Open_Entropy),
  na.rm = TRUE
)

df_label <- df_entropy %>%
  mutate(
    dist_to_diag = abs(Dep_Open_Entropy - Dep_Closed_Entropy),
    Data_clean = str_remove(Data, "^UD_"),
    Language = str_extract(Data_clean, "^[^-]+"),
    Treebank = str_extract(Data_clean, "(?<=-).*"),
    label = paste0(Language, " (", Treebank, ")")
  ) %>%
  slice_min(dist_to_diag, n = 4)

ggplot(df_entropy, aes(
  x = Dep_Closed_Entropy,
  y = Dep_Open_Entropy
)) +
  geom_point(
    size = 2,
    alpha = 0.7,
    color = "#6D8CD1"
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 0.7
  ) +geom_text_repel(
    data = df_label,
    aes(label = label),
    size = 3.5,
    box.padding = 1.25,
    point.padding = 0.95,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.color = "black",
    segment.size = 0.5,
    segment.alpha = 0.8,
    show.legend = FALSE
  ) + 
  scale_x_continuous(limits = lims) +
  scale_y_continuous(limits = lims) +
  coord_fixed() +
  labs(
    x = "Function-Word Dependency Entropy",
    y = "Content-Word Dependency Entropy"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )


library(tidyverse)
library(ggrepel)

df_fw <- read_tsv(
  "/Users/xiulinyang/Desktop/TODO/function_word/ud_stats/close_vs_open_count.tsv",
  show_col_types = FALSE
)

df_fw <- df_fw %>%
  mutate(
    treebank = str_extract(name, "(?<=-).*"),
    label = paste0(language, " (", treebank, ")")
  )

df_label_info <- df_fw %>%
  mutate(
    dist = abs(open_ratio - open_freq_ratio)
  ) %>%
  slice_min(dist, n = 4)

df_long <- df_fw %>%
  transmute(
    name,
    label,
    closed_type  = closed_ratio,
    closed_freq  = closed_freq_ratio,
    open_type    = open_ratio,
    open_freq    = open_freq_ratio
  ) %>%
  pivot_longer(
    cols = -c(name, label),
    names_to = c("class", ".value"),
    names_pattern = "(closed|open)_(type|freq)"
  ) %>%
  mutate(
    class = recode(
      class,
      "closed" = "Function words",
      "open"   = "Content words"
    )
  )

df_label <- df_long %>%
  filter(name %in% df_label_info$name, class%in% c("Content words", "Function words"))

ggplot(df_long, aes(
  x = type,
  y = freq,
  color = class
)) +
  geom_point(
    size = 2,
    alpha = 0.75
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "gray50",
    linewidth = 0.7
  ) +
  geom_text_repel(
    data = df_label,
    aes(label = label),
    size = 3.5,
    color = "black",
    box.padding = 0.75,
    point.padding = 0.5,
    max.overlaps = Inf,
    min.segment.length = 0,
    segment.color = "black",
    segment.size = 0.5,
    segment.alpha = 0.8,
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c(
      "Function words" = "#6D8CD1",
      "Content words"  = "#B64C4C"
    )
  ) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_fixed() +
  labs(
    x = "Type Ratio",
    y = "Token Frequency Ratio",
    color = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    legend.position = c(0.97, 0.03),
    legend.justification = c("right", "bottom"),
    legend.key = element_blank(),
    legend.text = element_text(size = 15),
    axis.title.x = element_text(size = 15),
    axis.title.y = element_text(size = 15)
  )
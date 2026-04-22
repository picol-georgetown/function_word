library(dplyr)
library(tidyr)
library(purrr)
library(readr)

L <- 0:11
H <- 0:11
seeds <- c(42, 53, 67)

# Shannon entropy
entropy_shannon <- function(x, base = 2) {
  x <- as.numeric(x)
  s <- sum(x)
  if (s <= 0) return(NA_real_)
  p <- x / s
  p <- p[p > 0]
  -sum(p * (log(p) / log(base)))
}

compute_entropy_one_seed <- function(seed) {
  
  df <- read_csv(
    paste0("/Users/xiulinyang/Desktop/TODO/function_word/function_head_", seed, ".csv"),
    show_col_types = FALSE
  ) %>%
    filter(freq > 0) %>%
    mutate(
      layer = as.numeric(layer),
      head  = as.numeric(head),
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
  
  df %>%
    group_by(condition) %>%
    group_modify(~{
      full_tbl <- tidyr::expand_grid(layer = L, head = H) %>%
        left_join(.x, by = c("layer", "head")) %>%
        mutate(freq = replace_na(freq, 0)) %>%
        arrange(layer, head)
      
      mat <- matrix(full_tbl$freq, nrow = 12, ncol = 12, byrow = TRUE)
      vec <- as.vector(mat)
      
      tibble(
        entropy_bits = entropy_shannon(vec, base = 2)
      )
    }) %>%
    ungroup() %>%
    mutate(seed = seed)
}

# ---- compute for all seeds ----
all_entropy <- map_dfr(seeds, compute_entropy_one_seed)

# ---- print each seed ----
for (s in seeds) {
  cat("\n========================\n")
  cat("Seed:", s, "\n")
  print(all_entropy %>% filter(seed == s))
}

# ---- average over seeds ----
avg_entropy <- all_entropy %>%
  group_by(condition) %>%
  summarise(
    mean_entropy = mean(entropy_bits),
    sd_entropy   = sd(entropy_bits),
    .groups = "drop"
  )

cat("\n========================\n")
cat("Average over seeds\n")
print(avg_entropy)
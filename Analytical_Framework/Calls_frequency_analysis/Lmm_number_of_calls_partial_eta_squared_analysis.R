# Number of calls - lmm, likelihood ratio tests, partial eta-squared
# Antonella Torrisi

rm(list = ls())

library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(effectsize)
library(readr)

options(contrasts = c("contr.sum", "contr.poly"))

# load data
setwd("/path/to/project")
data_path <- file.path("data", "full_dataset_vpa_ctrl.csv")
t <- read_csv(data_path)

# number of calls per chick, time bin, condition and cluster
calls_for_stats <- t %>%
  filter(!is.na(bin), !is.na(condition), !is.na(cluster_membership_2)) %>%
  group_by(id, bin, condition, cluster_membership_2) %>%
  summarise(total_calls = n(), .groups = "drop") %>%
  mutate(
    id                   = as.factor(id),
    bin                  = relevel(as.factor(bin), ref = "1"),
    condition            = as.factor(condition),
    cluster_membership_2 = as.factor(cluster_membership_2)
  )

alpha <- 0.05

fit <- function(f) lmer(f, data = calls_for_stats, REML = TRUE)
ml  <- function(f) lmer(f, data = calls_for_stats, REML = FALSE)

# partial eta-squared of one term from the type iii anova table
eta2_typeIII <- function(model, term) {
  es <- as.data.frame(eta_squared(anova(model, type = 3), partial = TRUE, alternative = "two.sided"))
  es$Eta2_partial[es$Parameter == term]
}

# likelihood ratio test and partial eta-squared
test_effect <- function(label, model, model_reduced, term) {
  a <- anova(model_reduced, model)
  data.frame(Effect = label, Chi_Square = a$Chisq[2], Df = a$Df[2],
             P_Value = a$`Pr(>Chisq)`[2], Partial_eta2 = eta2_typeIII(model, term))
}

main_effects <- "bin + condition + cluster_membership_2"
twoway_terms <- c("bin:condition", "bin:cluster_membership_2", "condition:cluster_membership_2")

build <- function(terms) {
  as.formula(paste("total_calls ~", paste(c(main_effects, terms), collapse = " + "), "+ (1 | id)"))
}

# full model: each effect removed with its higher-order interactions
model_full <- fit(total_calls ~ bin * condition * cluster_membership_2 + (1 | id))

tests_full <- rbind(
  test_effect("Cluster", model_full,
              fit(total_calls ~ bin * condition + (1 | id)), "cluster_membership_2"),
  test_effect("Condition", model_full,
              fit(total_calls ~ bin * cluster_membership_2 + (1 | id)), "condition"),
  test_effect("Temporal Bin", model_full,
              fit(total_calls ~ condition * cluster_membership_2 + (1 | id)), "bin"),
  test_effect("Condition x Cluster", model_full,
              fit(total_calls ~ bin + condition + cluster_membership_2 +
                    bin:condition + bin:cluster_membership_2 + (1 | id)), "condition:cluster_membership_2"),
  test_effect("Condition x Cluster x Bin", model_full,
              fit(total_calls ~ bin * condition + bin * cluster_membership_2 +
                    condition * cluster_membership_2 + (1 | id)), "bin:condition:cluster_membership_2")
)
print(tests_full)

# model selection: three-way interaction, then two-way interactions (least significant first)
# chi-square, p and aic of the models with and without each term are saved at every step
step_row <- function(step, term, m_with, m_without, decision) {
  a <- anova(m_without, m_with)
  data.frame(Step = step, Term_tested = term, Chi_Square = a$Chisq[2], Df = a$Df[2],
             P_Value = a$`Pr(>Chisq)`[2], AIC_with_term = AIC(m_with),
             AIC_without_term = AIC(m_without), delta_AIC = AIC(m_without) - AIC(m_with),
             Decision = decision)
}

m_3way <- ml(build(c(twoway_terms, "bin:condition:cluster_membership_2")))
m_2way <- ml(build(twoway_terms))
p3     <- anova(m_2way, m_3way)$`Pr(>Chisq)`[2]
selection_steps <- step_row(1, "bin:condition:cluster_membership_2", m_3way, m_2way,
                            ifelse(p3 < alpha, "retained", "dropped"))

if (p3 < alpha) {
  selected <- c(twoway_terms, "bin:condition:cluster_membership_2")
} else {
  selected <- twoway_terms
  step <- 1
  repeat {
    if (length(selected) == 0) break
    m_cur <- ml(build(selected))
    p <- sapply(selected, function(tt) anova(ml(build(setdiff(selected, tt))), m_cur)$`Pr(>Chisq)`[2])
    step  <- step + 1
    worst <- names(which.max(p))
    if (max(p) >= alpha) {
      selection_steps <- rbind(selection_steps,
                               step_row(step, worst, m_cur, ml(build(setdiff(selected, worst))), "dropped"))
      selected <- setdiff(selected, worst)
    } else {
      # all remaining two-way interactions significant
      for (tt in selected) {
        selection_steps <- rbind(selection_steps,
                                 step_row(step, tt, m_cur, ml(build(setdiff(selected, tt))), "retained"))
      }
      break
    }
  }
}
print(selection_steps)

model_mam <- fit(build(selected))
cat("minimum adequate model:", deparse(build(selected)), "\n")

# minimum adequate model: each factor removed with its interactions
drop_factor <- function(fac) {
  keep_main <- setdiff(c("bin", "condition", "cluster_membership_2"), fac)
  keep_int  <- selected[!grepl(fac, selected)]
  as.formula(paste("total_calls ~", paste(c(keep_main, keep_int), collapse = " + "), "+ (1 | id)"))
}

tests_mam <- rbind(
  test_effect("Cluster",      model_mam, fit(drop_factor("cluster_membership_2")), "cluster_membership_2"),
  test_effect("Condition",    model_mam, fit(drop_factor("condition")),            "condition"),
  test_effect("Temporal Bin", model_mam, fit(drop_factor("bin")),                  "bin")
)
for (tt in selected) {
  tests_mam <- rbind(tests_mam, test_effect(tt, model_mam, fit(build(setdiff(selected, tt))), tt))
}
print(tests_mam)

# posthoc
emm_int <- emmeans(model_mam, ~ condition * cluster_membership_2, lmer.df = "satterthwaite")
emm_bin <- emmeans(model_mam, ~ bin, lmer.df = "satterthwaite")

# conditions within cluster
posthoc_condition <- as.data.frame(summary(pairs(emm_int, by = "cluster_membership_2", adjust = "tukey")))
# clusters within condition
posthoc_cluster   <- as.data.frame(summary(pairs(emm_int, by = "condition", adjust = "tukey")))
# time bins
posthoc_bin       <- as.data.frame(summary(pairs(emm_bin, adjust = "tukey")))

print(posthoc_condition)
print(posthoc_cluster)
print(posthoc_bin)

write.csv(tests_full,        "Number_calls_results_LRT_partial_eta2_full_model.csv",    row.names = FALSE)
write.csv(selection_steps,   "Number_calls_model_selection_steps.csv",                  row.names = FALSE)
write.csv(tests_mam,         "Number_calls_results_LRT_partial_eta2_minimum_model.csv", row.names = FALSE)
write.csv(posthoc_condition, "Number_calls_posthoc_condition_within_cluster.csv",        row.names = FALSE)
write.csv(posthoc_cluster,   "Number_calls_posthoc_cluster_within_condition.csv",        row.names = FALSE)
write.csv(posthoc_bin,       "Number_calls_posthoc_time_bin.csv",                        row.names = FALSE)

# package versions
writeLines(capture.output(sessionInfo()), "sessionInfo_number_of_calls.txt")

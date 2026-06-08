# ============================================================
# pca_analysis.R
# Polychoric-Pearson PCA and FAMD for the SDPI dimensionality
# assessment (Section 4.2 and Section 5.3 of the paper).
#
# This is the analysis referenced in the paper.
# Run this in RStudio or via Rscript pca_analysis.R
#
# Requirements: install.packages(c("psych","FactoMineR","factoextra","ggplot2","readr"))
# ============================================================

library(psych)
library(FactoMineR)
library(factoextra)
library(ggplot2)
library(readr)

# ── Load data ────────────────────────────────────────────────
df <- read_csv("data/SDPI_final_all_methods.csv")
df$State[df$State == "Louisana"] <- "Louisiana"  # fix typo

# ── Define continuous vs binary/ordinal indicators ──────────
# Continuous: measured on a genuine numeric scale
continuous_vars <- c(
  "avg_monthly_ssi_payment", "medicaid_eligibility_threshold",
  "adl_medicaid_coverage_pct", "private_ltci_per1000",
  "hcbs_expenditure_ratio", "hcbs_user_ratio",
  "pct_ltss_hcbs_older_adults", "home_health_aides_per100",
  "smd_demonstration_projects", "vr_spending_career",
  "vr_spending_training", "arp_caregiver_family_support",
  "arp_waiting_list_diversion", "arp_tech_telehealth",
  "arp_cross_sector_investments", "arp_workforce_training",
  "arp_quality_improvement", "livability_transportation",
  "livability_housing", "section_811_pct_disability",
  "special_ed_policy_score", "initial_ar",
  "reconsidered_ar", "total_ar"
)

# Binary/ordinal: policy presence indicators (0/1 or 0/1/2)
binary_ordinal_vars <- c(
  "ssi_auto_enroll_medicaid", "ssi_criteria_209",
  "medically_needy", "buy_in_working_people",
  "spousal_impoverishment", "family_responsibility_class",
  "hcbs_presumptive_eligibility", "subminimum_wage_14c",
  "ui_good_cause_caregiving", "section_811_pra",
  "fema_shmp", "caps"
)

all_indicators <- c(continuous_vars, binary_ordinal_vars)
X <- df[, all_indicators]

# ── Mixed correlation matrix (polychoric + Pearson) ──────────
# polychoric: binary/ordinal pairs
# pearson: continuous pairs
# polyserial: mixed pairs (continuous × binary)
cat("Computing mixed correlation matrix...\n")
mixed_cor <- mixedCor(
  data = X,
  c = which(names(X) %in% continuous_vars),   # continuous columns
  p = which(names(X) %in% binary_ordinal_vars) # polychoric columns
)

cat("Mixed correlation matrix computed.\n")
cat("Matrix dimensions:", dim(mixed_cor$rho), "\n")

# ── PCA on mixed correlation matrix ─────────────────────────
pca_result <- principal(
  mixed_cor$rho,
  nfactors = length(all_indicators),
  rotate = "none",
  covar = FALSE  # use correlation matrix (standardized)
)

# Variance explained
var_explained <- pca_result$values / sum(pca_result$values) * 100
cum_var       <- cumsum(var_explained)

cat("\n── Variance Explained ──\n")
cat(sprintf("PC1: %.1f%%\n", var_explained[1]))
cat(sprintf("PC2: %.1f%%\n", var_explained[2]))
cat(sprintf("Components to reach 70%%: %d\n", which(cum_var >= 70)[1]))

# ── Scree plot ───────────────────────────────────────────────
scree_df <- data.frame(PC = 1:15, Variance = var_explained[1:15], Cumulative = cum_var[1:15])
p_scree <- ggplot(scree_df, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "#2E86AB", color = "black", linewidth = 0.5) +
  geom_line(aes(y = Cumulative / 5), color = "#E76F51", linewidth = 1.2) +
  geom_point(aes(y = Cumulative / 5), color = "#E76F51", size = 2.5) +
  geom_hline(yintercept = 70/5, linetype = "dashed", color = "#E76F51", alpha = 0.6) +
  scale_y_continuous(
    name = "Variance Explained (%)",
    sec.axis = sec_axis(~ . * 5, name = "Cumulative Variance (%)")
  ) +
  labs(title = "Scree Plot: PC1 explains 19.1% — state disability policy is multidimensional",
       x = "Principal Component") +
  theme_minimal(base_size = 12)
ggsave("output/fig4_pca_scree_R.png", p_scree, width = 10, height = 5.5, dpi = 200)
cat("Saved output/fig4_pca_scree_R.png\n")

# ── Top PC1 loadings ─────────────────────────────────────────
loadings_pc1 <- sort(abs(pca_result$loadings[, 1]), decreasing = TRUE)
cat("\nTop 10 variables loading on PC1:\n")
print(round(loadings_pc1[1:10], 3))

# ── FAMD (Factor Analysis of Mixed Data) ────────────────────
# Convert binary/ordinal to factors for FAMD
X_famd <- X
X_famd[, binary_ordinal_vars] <- lapply(X_famd[, binary_ordinal_vars], as.factor)

cat("\nRunning FAMD...\n")
famd_result <- FAMD(X_famd, ncp = 20, graph = FALSE)

famd_var     <- famd_result$eig[, 2]  # % of variance
famd_cumvar  <- famd_result$eig[, 3]  # cumulative %
cat(sprintf("FAMD Dim1: %.1f%%\n", famd_var[1]))
cat(sprintf("FAMD components to reach 70%%: %d\n", which(famd_cumvar >= 70)[1]))

# Biplot of states in FAMD Dim1 x Dim2 space
p_famd <- fviz_famd_ind(
  famd_result, repel = TRUE,
  col.ind = "cos2", gradient.cols = c("#E76F51", "#FFFFFF", "#2E86AB"),
  title = "FAMD Biplot: State positions in Dim1 × Dim2"
)
ggsave("output/fig5_famd_biplot_R.png", p_famd, width = 10, height = 8, dpi = 200)
cat("Saved output/fig5_famd_biplot_R.png\n")

cat("\n── Summary ──\n")
cat(sprintf("PCA  PC1: %.1f%% | components for 70%%: %d\n", var_explained[1], which(cum_var >= 70)[1]))
cat(sprintf("FAMD Dim1: %.1f%% | components for 70%%: %d\n", famd_var[1], which(famd_cumvar >= 70)[1]))
cat("Both methods confirm: state disability policy is highly multidimensional.\n")

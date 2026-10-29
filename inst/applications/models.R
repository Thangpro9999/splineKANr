# Measurement and structural models used in the application scripts.

# ECSI model of seminr's `mobi` data (mobile phone customers, n = 250),
# as in the seminr ECSI demonstration.
mobi_model <- list(
  blocks = list(
    Image = paste0("IMAG", 1:5), Expectation = paste0("CUEX", 1:3),
    Quality = paste0("PERQ", 1:7), Value = paste0("PERV", 1:2),
    Satisfaction = paste0("CUSA", 1:3), Complaints = "CUSCO",
    Loyalty = paste0("CUSL", 1:3)),
  paths = rbind(
    c("Image", "Expectation"), c("Image", "Satisfaction"),
    c("Image", "Loyalty"), c("Expectation", "Quality"),
    c("Expectation", "Value"), c("Expectation", "Satisfaction"),
    c("Quality", "Value"), c("Quality", "Satisfaction"),
    c("Value", "Satisfaction"), c("Satisfaction", "Complaints"),
    c("Satisfaction", "Loyalty"), c("Complaints", "Loyalty"))
)

# ECSI model of plspm's `satisfaction` data (n = 250), as in the plspm
# documentation (Sanchez 2013, PLS Path Modeling with R, ch. 4).
satisfaction_model <- list(
  blocks = list(
    IMAG = paste0("imag", 1:5), EXPE = paste0("expe", 1:5),
    QUAL = paste0("qual", 1:5), VAL = paste0("val", 1:4),
    SAT = paste0("sat", 1:4), LOY = paste0("loy", 1:4)),
  paths = rbind(
    c("IMAG", "EXPE"), c("IMAG", "SAT"), c("IMAG", "LOY"),
    c("EXPE", "QUAL"), c("EXPE", "VAL"), c("EXPE", "SAT"),
    c("QUAL", "VAL"), c("QUAL", "SAT"), c("VAL", "SAT"), c("SAT", "LOY"))
)

# Extended corporate reputation model of seminr's `corp_rep_data` (n = 344;
# Hair et al. 2022, chapters 5-6): formative QUAL, PERF, CSOR, ATTR (mode B),
# reflective COMP, LIKE, CUSL, single-item CUSA. Missing values are coded
# -99 and are mean-replaced (the book's and seminr's default); in cross-
# validation the training-fold means are used.
corp_rep_model <- list(
  blocks = list(
    QUAL = paste0("qual_", 1:8), PERF = paste0("perf_", 1:5),
    CSOR = paste0("csor_", 1:5), ATTR = paste0("attr_", 1:3),
    COMP = paste0("comp_", 1:3), LIKE = paste0("like_", 1:3),
    CUSA = "cusa", CUSL = paste0("cusl_", 1:3)),
  paths = rbind(
    cbind(rep(c("QUAL", "PERF", "CSOR", "ATTR"), 2),
          rep(c("COMP", "LIKE"), each = 4)),
    cbind(rep(c("COMP", "LIKE"), 2), rep(c("CUSA", "CUSL"), each = 2)),
    c("CUSA", "CUSL")),
  modes = c(QUAL = "B", PERF = "B", CSOR = "B", ATTR = "B")
)
na_code <- function(d, code = -99) {
  d[d == code] <- NA
  d
}
mean_impute <- function(d, ref = d) {
  for (v in names(d)) {
    if (anyNA(d[[v]])) d[[v]][is.na(d[[v]])] <- mean(ref[[v]], na.rm = TRUE)
  }
  d
}

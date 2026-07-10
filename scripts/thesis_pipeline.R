# ============================================================
# Thesis result visuals in R: "Judging the Judge"
# ------------------------------------------------------------
# This script reproduces the FINAL, CORRECTED statistics and
# generates the six figures used in the thesis (Figures 2-7).
#
# NOTE ON THE NUMBERS: the counts below are the final, corrected
# values reported in Table 2 of the thesis. They were produced
# and verified by the Python/scipy pipeline (see scripts/ and
# make_rq1_charts.py). This R script is the presentation/plotting
# layer for those verified results, not a separate analysis.
#
# DeepSeek V3.2 is at full n=999 (both batches combined).
# RQ2 has the 11 genuine cases (10 unclear, 1 neutral).
# Outputs: 6 PNG figures + rq1_summary_table_4models_FINAL_R.csv
# ============================================================
if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)
# ============================================================
# 1. RQ1: Hallucination rate by frame, ALL FOUR MODELS, FINAL
# ============================================================
gpt4o <- matrix(c(996,3, 997,2, 998,1), nrow=3, byrow=TRUE,
                dimnames=list(c("neutral","positive","negative"), c("supported","hallucinated")))
gpt4o_test <- chisq.test(gpt4o, correct=FALSE)
deepseek <- matrix(c(997,2, 998,1, 998,1), nrow=3, byrow=TRUE,
                   dimnames=list(c("neutral","positive","negative"), c("supported","hallucinated")))
deepseek_test <- chisq.test(deepseek, correct=FALSE)
llama31 <- matrix(c(98,1, 99,0, 98,1), nrow=3, byrow=TRUE,
                  dimnames=list(c("neutral","positive","negative"), c("supported","hallucinated")))
llama31_test <- chisq.test(llama31, correct=FALSE)
llama33 <- matrix(c(99,0, 99,0, 98,1), nrow=3, byrow=TRUE,
                  dimnames=list(c("neutral","positive","negative"), c("supported","hallucinated")))
llama33_test <- chisq.test(llama33, correct=FALSE)
cat("=== RQ1 chi-square tests, all four models (FINAL) ===\n")
cat(sprintf("GPT-4o-mini:    chi2=%.2f, p=%.3f\n", gpt4o_test$statistic, gpt4o_test$p.value))
cat(sprintf("DeepSeek V3.2:  chi2=%.2f, p=%.3f\n", deepseek_test$statistic, deepseek_test$p.value))
cat(sprintf("Llama 3.1 70B:  chi2=%.2f, p=%.3f\n", llama31_test$statistic, llama31_test$p.value))
cat(sprintf("Llama 3.3 70B:  chi2=%.2f, p=%.3f\n\n", llama33_test$statistic, llama33_test$p.value))
rq1_df <- data.frame(
  model = rep(c("GPT-4o-mini\n(n=999)", "DeepSeek V3.2\n(n=999)", "Llama 3.1 70B\n(n=99)", "Llama 3.3 70B\n(n=99)"), each=3),
  frame = rep(c("Neutral","Positive","Negative"), times=4),
  rate = c(0.30,0.20,0.10,  0.20,0.10,0.10,  1.01,0.00,1.01,  0.00,0.00,1.01)
)
rq1_df$frame <- factor(rq1_df$frame, levels=c("Neutral","Positive","Negative"))
rq1_df$model <- factor(rq1_df$model, levels=c("GPT-4o-mini\n(n=999)", "DeepSeek V3.2\n(n=999)", "Llama 3.1 70B\n(n=99)", "Llama 3.3 70B\n(n=99)"))
p1 <- ggplot(rq1_df, aes(x=frame, y=rate, fill=frame)) +
  geom_col(width=0.6) +
  facet_wrap(~model, nrow=1) +
  geom_text(aes(label=sprintf("%.2f%%", rate)), vjust=-0.5, size=3.2) +
  scale_fill_manual(values=c("Neutral"="#5B7DB1","Positive"="#D98C4A","Negative"="#6FA287")) +
  labs(title="RQ1: Hallucination rate by frame, corrected judge",
       subtitle="No statistically significant difference across frames for any of the four models",
       x=NULL, y="Hallucination rate (%)") +
  theme_minimal(base_size=12) +
  theme(legend.position="none", plot.title=element_text(face="bold"), strip.text=element_text(face="bold", size=9))
ggsave("rq1_hallucination_rate_4models_FINAL_R.png", p1, width=11, height=5, dpi=200)
cat("Saved: rq1_hallucination_rate_4models_FINAL_R.png\n\n")
# ============================================================
# 2. Judge correction before/after - GPT-4o-mini ONLY
# ============================================================
correction_df <- data.frame(
  frame = rep(c("Neutral","Positive","Negative"), times=2),
  stage = rep(c("Before correction","After correction"), each=3),
  rate = c(2.10,3.71,1.90,  0.30,0.20,0.10)
)
correction_df$frame <- factor(correction_df$frame, levels=c("Neutral","Positive","Negative"))
correction_df$stage <- factor(correction_df$stage, levels=c("Before correction","After correction"))
p2 <- ggplot(correction_df, aes(x=frame, y=rate, fill=stage)) +
  geom_col(position=position_dodge(width=0.7), width=0.6) +
  geom_text(aes(label=sprintf("%.2f%%", rate)), position=position_dodge(width=0.7), vjust=-0.5, size=3.2) +
  scale_fill_manual(values=c("Before correction"="#C75D4D","After correction"="#4D8FC7")) +
  labs(title="Effect of judge correction on hallucination rate",
       subtitle="GPT-4o-mini, n=999 per frame (the model used to discover and fix the bug)",
       x=NULL, y="Hallucination rate (%)", fill=NULL) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"), legend.position="top")
ggsave("judge_correction_effect_R.png", p2, width=8, height=5, dpi=200)
cat("Saved: judge_correction_effect_R.png\n\n")
# ============================================================
# 3. RQ2: Directional coding donut, FINAL 11 cases (10 unclear, 1 neutral)
# ============================================================
rq2_df <- data.frame(category=c("Unclear","Neutral"), count=c(10,1))
rq2_df$fraction <- rq2_df$count / sum(rq2_df$count)
rq2_df$ymax <- cumsum(rq2_df$fraction)
rq2_df$ymin <- c(0, head(rq2_df$ymax, n=-1))
rq2_df$label <- paste0(rq2_df$category, "\n", rq2_df$count, " of 11")
p3 <- ggplot(rq2_df, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=category)) +
  geom_rect(color="white") +
  coord_polar(theta="y") +
  xlim(c(2,4)) +
  geom_text(aes(x=3.5, y=(ymin+ymax)/2, label=label), size=3.5, color="white") +
  scale_fill_manual(values=c("Unclear"="#888780","Neutral"="#1D9E75")) +
  labs(title="RQ2: direction of the 11 genuine hallucination cases") +
  theme_void(base_size=12) +
  theme(plot.title=element_text(face="bold", hjust=0.5), legend.position="none")
ggsave("rq2_donut_11cases_FINAL_R.png", p3, width=6, height=6, dpi=200)
cat("Saved: rq2_donut_11cases_FINAL_R.png\n\n")
# ============================================================
# 4. RQ3: Entropy before/after - GPT-4o-mini ONLY
# ============================================================
entropy_df <- data.frame(
  stage = rep(c("Before correction","After correction"), each=2),
  outcome = rep(c("Supported","Hallucinated"), times=2),
  entropy = c(0.068,0.127, 0.070,0.061)
)
entropy_df$stage <- factor(entropy_df$stage, levels=c("Before correction","After correction"))
p4 <- ggplot(entropy_df, aes(x=stage, y=entropy, color=outcome, group=outcome)) +
  geom_line(linewidth=1) +
  geom_point(size=4) +
  geom_text(aes(label=entropy), vjust=-1, size=3.2, show.legend=FALSE) +
  scale_color_manual(values=c("Supported"="#5B7DB1","Hallucinated"="#C75D4D")) +
  labs(title="RQ3: mean entropy, supported vs hallucinated responses",
       subtitle="GPT-4o-mini only (logprobs unavailable from the other 3 providers)",
       x=NULL, y="Mean token entropy", color=NULL) +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"), legend.position="top")
ggsave("rq3_entropy_lines_R.png", p4, width=7, height=5, dpi=200)
cat("Saved: rq3_entropy_lines_R.png\n\n")
# ============================================================
# 5. EU pilot funnel
# ============================================================
eu_df <- data.frame(lean=c("Center (6 outlets)","Right (Breitbart)","Left (Jacobin)"),
                    supported=c(18,3,3), total=c(18,3,3))
eu_df$lean <- factor(eu_df$lean, levels=rev(eu_df$lean))
p5 <- ggplot(eu_df, aes(x=lean, y=supported)) +
  geom_col(fill="#1D9E75", width=0.55) +
  geom_text(aes(label=paste0(supported," / ",total," supported")), hjust=-0.1, size=3.4) +
  coord_flip() +
  scale_y_continuous(limits=c(0,22), expand=c(0,0)) +
  labs(title="EU pilot: all 24 responses supported, by source lean", x=NULL, y="Responses") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold"), axis.text.x=element_blank(), panel.grid=element_blank())
ggsave("eu_pilot_funnel_R.png", p5, width=8, height=4, dpi=200)
cat("Saved: eu_pilot_funnel_R.png\n\n")
# ============================================================
# 6. Harder-task exploratory probe
# ============================================================
probe_overall <- data.frame(
  task = c("Single-article task\n(n=3,591 responses)", "Multi-article probe\n(n=45 responses)"),
  rate = c(0.17, 2.22)
)
probe_overall$task <- factor(probe_overall$task, levels = probe_overall$task)
p6 <- ggplot(probe_overall, aes(x=task, y=rate, fill=task)) +
  geom_col(width=0.55) +
  geom_text(aes(label=sprintf("%.2f%%", rate)), vjust=-0.5, size=4) +
  scale_fill_manual(values=c("#378ADD","#BA7517")) +
  labs(title="Exploratory probe: hallucination rate, easy vs harder task",
       subtitle="Combining 3 articles into multi-fact questions, n=45, descriptive only",
       x=NULL, y="Hallucination rate (%)") +
  theme_minimal(base_size=12) +
  theme(legend.position="none", plot.title=element_text(face="bold"))
ggsave("harder_task_probe_overall_R.png", p6, width=7, height=5, dpi=200)
cat("Saved: harder_task_probe_overall_R.png\n\n")
# ============================================================
# FINAL summary table, ALL FOUR MODELS
# ============================================================
summary_df <- data.frame(
  Model = c("GPT-4o-mini","DeepSeek V3.2","Llama 3.1 70B","Llama 3.3 70B"),
  N_per_frame = c(999,999,99,99),
  Chi_square = c(round(gpt4o_test$statistic,2), round(deepseek_test$statistic,2),
                 round(llama31_test$statistic,2), round(llama33_test$statistic,2)),
  P_value = c(round(gpt4o_test$p.value,3), round(deepseek_test$p.value,3),
              round(llama31_test$p.value,3), round(llama33_test$p.value,3)),
  Significant = c("No","No","No","No")
)
print(summary_df)
write.csv(summary_df, "rq1_summary_table_4models_FINAL_R.csv", row.names=FALSE)
cat("\nSaved: rq1_summary_table_4models_FINAL_R.csv\n")
cat("\nAll six figures + summary table saved to working directory.\n")

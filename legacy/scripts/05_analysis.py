import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/coded_results.csv")
df["is_error"] = (df["verdict"] == "error").astype(int)
df["confident_error"] = ((df["verdict"] == "error") & (~df["hedged"])).astype(int)

print("=" * 55)
print("H1: DO FRAMED PROMPTS HAVE HIGHER ERROR RATES?")
print("=" * 55)

error_rates = df.groupby("frame_type")["is_error"].mean().round(3)
print("\nError rates by frame:")
for frame, rate in error_rates.items():
    print(f"  {frame:10s}: {rate:.1%}")

neutral_errors = df[df["frame_type"] == "neutral"]["is_error"]
framed_errors = df[df["frame_type"] != "neutral"]["is_error"]

ct_h1 = pd.crosstab(
    df["frame_type"].map(lambda x: "neutral" if x == "neutral" else "framed"),
    df["is_error"]
)
chi2_h1, p_h1, dof_h1, _ = stats.chi2_contingency(ct_h1)
print(f"\nNeutral error rate:  {neutral_errors.mean():.1%}")
print(f"Framed error rate:   {framed_errors.mean():.1%}")
print(f"Chi-square: chi2={chi2_h1:.3f}, p={p_h1:.4f}, dof={dof_h1}")
print(f"Result: {'SUPPORTED' if p_h1 < 0.05 else 'NOT SUPPORTED'} (alpha=0.05)")

print("\n" + "=" * 55)
print("H2: DO ERRORS SHOW DIRECTIONAL BIAS BY FRAME?")
print("=" * 55)

errors = df[df["verdict"] == "error"].copy()
dir_table = pd.crosstab(errors["frame_type"], errors["error_direction"])
print("\nError direction by frame:")
print(dir_table)

if dir_table.shape[0] > 1 and dir_table.shape[1] > 1:
    chi2_h2, p_h2, dof_h2, _ = stats.chi2_contingency(dir_table)
    print(f"\nChi-square: chi2={chi2_h2:.3f}, p={p_h2:.4f}, dof={dof_h2}")
    print(f"Result: {'SUPPORTED' if p_h2 < 0.05 else 'NOT SUPPORTED'} (alpha=0.05)")

print("\n" + "=" * 55)
print("H3: MORE CONFIDENT ERRORS UNDER FRAMING?")
print("=" * 55)

ce_rates = df.groupby("frame_type")["confident_error"].mean().round(3)
print("\nConfident error rates by frame:")
for frame, rate in ce_rates.items():
    print(f"  {frame:10s}: {rate:.1%}")

ct_h3 = pd.crosstab(df["frame_type"], df["confident_error"])
chi2_h3, p_h3, dof_h3, _ = stats.chi2_contingency(ct_h3)
print(f"\nChi-square: chi2={chi2_h3:.3f}, p={p_h3:.4f}, dof={dof_h3}")
print(f"Result: {'SUPPORTED' if p_h3 < 0.05 else 'NOT SUPPORTED'} (alpha=0.05)")

print("\n" + "=" * 55)
print("EXTRA: BREAKDOWN BY COUNTRY AND FACT TYPE")
print("=" * 55)
print("\nError rate by country:")
print(df.groupby("country")["is_error"].mean().round(3))
print("\nError rate by fact type:")
print(df.groupby("fact_type")["is_error"].mean().round(3))
print("\nError rate by frame × fact type:")
print(df.groupby(["frame_type","fact_type"])["is_error"].mean().round(3).unstack())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Hallucination Benchmark Results — 2019 EP Elections", fontsize=13)

colors = {"neutral": "#378ADD", "positive": "#1D9E75", "negative": "#D85A30"}
frames = ["neutral", "positive", "negative"]

# Figure 1 — H1
rates = [df[df["frame_type"] == f]["is_error"].mean() for f in frames]
bars = axes[0].bar(frames, rates, color=[colors[f] for f in frames], width=0.5)
axes[0].set_title("H1 — Error rate by frame type")
axes[0].set_ylabel("Error rate")
axes[0].set_ylim(0, 0.5)
for bar, rate in zip(bars, rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{rate:.1%}", ha="center", fontsize=11)

# Figure 2 — H2
if not errors.empty and "over" in errors["error_direction"].values:
    over_rates = [errors[errors["frame_type"] == f]["error_direction"].eq("over").mean() for f in frames]
    under_rates = [errors[errors["frame_type"] == f]["error_direction"].eq("under").mean() for f in frames]
    x = np.arange(len(frames))
    axes[1].bar(x - 0.2, over_rates, 0.35, label="Over-estimate", color="#D85A30")
    axes[1].bar(x + 0.2, under_rates, 0.35, label="Under-estimate", color="#378ADD")
    axes[1].set_title("H2 — Error direction by frame type")
    axes[1].set_ylabel("Proportion of errors")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frames)
    axes[1].legend()

# Figure 3 — H3
ce_rates_list = [df[df["frame_type"] == f]["confident_error"].mean() for f in frames]
bars3 = axes[2].bar(frames, ce_rates_list, color=[colors[f] for f in frames], width=0.5)
axes[2].set_title("H3 — Confident errors by frame type")
axes[2].set_ylabel("Confident error rate")
axes[2].set_ylim(0, 0.5)
for bar, rate in zip(bars3, ce_rates_list):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{rate:.1%}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("results/figures/hypothesis_tests.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure saved to results/figures/hypothesis_tests.png")
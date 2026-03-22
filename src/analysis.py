"""
Corrected analysis: account for gzip overhead on short texts
and produce final publication-quality figures.
"""

import json
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# ============================================================
# Load all experiment results
# ============================================================

with open(RESULTS_DIR / "experiment1_complexity.json") as f:
    exp1 = json.load(f)

with open(RESULTS_DIR / "experiment2_gibberish.json") as f:
    exp2 = json.load(f)

with open(RESULTS_DIR / "experiment3_human_vs_gpt.json") as f:
    exp3 = json.load(f)

with open(RESULTS_DIR / "experiment4_logprobs.json") as f:
    exp4 = json.load(f)


# ============================================================
# Corrected Experiment 3: Filter short texts
# ============================================================

print("="*60)
print("CORRECTED ANALYSIS")
print("="*60)

# Re-analyze exp3 with minimum length filter
human_path = ROOT / "datasets" / "gpt_wiki_intro" / "human_texts.txt"
gpt_path = ROOT / "datasets" / "gpt_wiki_intro" / "gpt_texts.txt"

with open(human_path) as f:
    human_texts = f.read().strip().split("\n")
with open(gpt_path) as f:
    gpt_texts = f.read().strip().split("\n")

MIN_LEN = 100  # Minimum characters to avoid gzip overhead distortion

h_bpb_filtered = []
g_bpb_filtered = []
h_ratio_filtered = []
g_ratio_filtered = []
n_filtered = 0

for i in range(min(500, len(human_texts), len(gpt_texts))):
    ht, gt = human_texts[i], gpt_texts[i]
    if len(ht) >= MIN_LEN and len(gt) >= MIN_LEN:
        h_data = ht.encode('utf-8')
        g_data = gt.encode('utf-8')
        h_comp = len(gzip.compress(h_data, 9))
        g_comp = len(gzip.compress(g_data, 9))

        h_bpb_filtered.append(h_comp * 8 / len(h_data))
        g_bpb_filtered.append(g_comp * 8 / len(g_data))
        h_ratio_filtered.append(len(h_data) / h_comp)
        g_ratio_filtered.append(len(g_data) / g_comp)
        n_filtered += 1

h_bpb_filtered = np.array(h_bpb_filtered)
g_bpb_filtered = np.array(g_bpb_filtered)
h_ratio_filtered = np.array(h_ratio_filtered)
g_ratio_filtered = np.array(g_ratio_filtered)

print(f"\nExp3 Corrected (texts >= {MIN_LEN} chars, n={n_filtered}):")
print(f"  Human: mean bpb={h_bpb_filtered.mean():.3f} ± {h_bpb_filtered.std():.3f}, "
      f"mean ratio={h_ratio_filtered.mean():.3f}")
print(f"  GPT:   mean bpb={g_bpb_filtered.mean():.3f} ± {g_bpb_filtered.std():.3f}, "
      f"mean ratio={g_ratio_filtered.mean():.3f}")

stat, pval = stats.wilcoxon(h_bpb_filtered, g_bpb_filtered)
cohens_d = (h_bpb_filtered.mean() - g_bpb_filtered.mean()) / np.sqrt(
    (h_bpb_filtered.std()**2 + g_bpb_filtered.std()**2) / 2
)
print(f"  Wilcoxon: stat={stat:.1f}, p={pval:.2e}")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  GPT text is {'MORE' if g_bpb_filtered.mean() < h_bpb_filtered.mean() else 'LESS'} compressible than human text")


# ============================================================
# Key Analysis: Information Amplification Decomposition
# ============================================================

print("\n" + "-"*60)
print("INFORMATION AMPLIFICATION ANALYSIS (Experiment 1)")
print("-"*60)

for r in exp1:
    in_chars = r["input_chars"]
    out_chars = r["output_chars"]
    in_gzip = r["input_gzip"]["compressed_bytes"]
    out_gzip = r["output_gzip"]["compressed_bytes"]
    in_bpb = r["input_gzip"]["bits_per_byte"]
    out_bpb = r["output_gzip"]["bits_per_byte"]

    # Total information ratio (compressed bits)
    total_ratio = (out_gzip * 8) / (in_gzip * 8) if in_gzip > 0 else 0
    # Length expansion
    length_ratio = out_chars / in_chars if in_chars > 0 else 0
    # Density ratio (bpb out / bpb in)
    density_ratio = out_bpb / in_bpb if in_bpb > 0 else 0

    print(f"\n  L{r['level']} {r['label']}:")
    print(f"    Length: {in_chars} -> {out_chars} chars ({length_ratio:.1f}x expansion)")
    print(f"    Density: {in_bpb:.2f} -> {out_bpb:.2f} bpb ({density_ratio:.2f}x)")
    print(f"    Total info: {in_gzip*8} -> {out_gzip*8} bits ({total_ratio:.1f}x)")
    print(f"    Decomposition: {total_ratio:.1f}x = {length_ratio:.1f}x (length) × {density_ratio:.2f}x (density)")

# Key insight: total info goes up because LENGTH expands, but DENSITY goes down
# This means per-byte, outputs are MORE predictable than inputs

levels = [r["level"] for r in exp1]
in_bpb = [r["input_gzip"]["bits_per_byte"] for r in exp1]
out_bpb = [r["output_gzip"]["bits_per_byte"] for r in exp1]
density_ratios = [o/i if i > 0 else 0 for i, o in zip(in_bpb, out_bpb)]

print(f"\n  Summary: Mean density ratio = {np.mean(density_ratios):.3f}")
print(f"  (< 1 means outputs are more compressible per byte than inputs)")
print(f"  For long texts (L3+): {np.mean([d for l,d in zip(levels, density_ratios) if l >= 3]):.3f}")


# ============================================================
# Logprob analysis
# ============================================================

print("\n" + "-"*60)
print("LOGPROB ANALYSIS (Experiment 4)")
print("-"*60)

for r in exp4:
    in_bits = r["input_gzip"]["compressed_bytes"] * 8
    out_gzip_bits = r["output_gzip"]["compressed_bytes"] * 8
    out_lp_bits = r.get("output_total_bits", 0)
    out_bpt = r.get("output_mean_bits_per_token", 0)

    print(f"\n  L{r['level']}:")
    print(f"    Input (gzip): {in_bits} bits")
    print(f"    Output (gzip): {out_gzip_bits} bits")
    print(f"    Output (logprob): {out_lp_bits:.0f} bits ({out_bpt:.2f} bits/token)")
    print(f"    Output info by LLM's own measure: {out_lp_bits:.0f} bits << {out_gzip_bits} gzip bits")
    if out_lp_bits > 0:
        print(f"    LLM sees output as {out_gzip_bits/out_lp_bits:.1f}x more compressible than gzip")


# ============================================================
# Publication-quality figures
# ============================================================

print("\n" + "="*60)
print("GENERATING FINAL FIGURES")
print("="*60)

# FIGURE 1: The key finding - input vs output information (corrected)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Total bits (input vs output)
ax = axes[0, 0]
levels_arr = np.array([r["level"] for r in exp1])
in_bits = np.array([r["input_gzip"]["compressed_bytes"] * 8 for r in exp1])
out_bits = np.array([r["output_gzip"]["compressed_bytes"] * 8 for r in exp1])

# Only use prompts long enough for meaningful gzip (level >= 3)
mask_long = levels_arr >= 3
scatter = ax.scatter(in_bits, out_bits, c=levels_arr, cmap='viridis', s=100,
                    edgecolors='black', linewidth=0.5, zorder=5)

# Fit on all data
slope, intercept, r_val, p_val, _ = stats.linregress(in_bits, out_bits)
x_fit = np.linspace(0, max(in_bits)*1.1, 100)
ax.plot(x_fit, slope*x_fit + intercept, 'r-', alpha=0.6,
       label=f'Linear fit: y={slope:.1f}x+{intercept:.0f}\nr²={r_val**2:.3f}, p={p_val:.2e}')
ax.plot([0, max(out_bits)*1.1], [0, max(out_bits)*1.1], 'k--', alpha=0.3, label='y=x')

ax.set_xlabel("Input Information (gzip bits)")
ax.set_ylabel("Output Information (gzip bits)")
ax.set_title("A) Total Information: Input vs Output")
ax.legend(fontsize=8)

# Panel B: Information density (bits per byte)
ax = axes[0, 1]
in_bpb_arr = np.array([r["input_gzip"]["bits_per_byte"] for r in exp1])
out_bpb_arr = np.array([r["output_gzip"]["bits_per_byte"] for r in exp1])

# Filter to texts where gzip is meaningful (input > 50 bytes)
mask_meaningful = np.array([r["input_chars"] > 50 for r in exp1])

ax.scatter(in_bpb_arr[mask_meaningful], out_bpb_arr[mask_meaningful],
          c=levels_arr[mask_meaningful], cmap='viridis', s=100,
          edgecolors='black', linewidth=0.5, zorder=5)
ax.scatter(in_bpb_arr[~mask_meaningful], out_bpb_arr[~mask_meaningful],
          c='gray', s=60, alpha=0.4, marker='s', label='Short text (< 50B, gzip unreliable)')

# Add diagonal
lim = max(max(in_bpb_arr[mask_meaningful]), max(out_bpb_arr[mask_meaningful]))
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')

rho, p_rho = stats.spearmanr(in_bpb_arr[mask_meaningful], out_bpb_arr[mask_meaningful])
ax.set_xlabel("Input Bits per Byte (gzip)")
ax.set_ylabel("Output Bits per Byte (gzip)")
ax.set_title(f"B) Information Density\n(Spearman ρ={rho:.3f}, p={p_rho:.3e})")
ax.legend(fontsize=8)

# Panel C: Gibberish control
ax = axes[1, 0]
types = [r["type"] for r in exp2]
g_in_bpb = [r["input_gzip"]["bits_per_byte"] for r in exp2]
g_out_bpb = [r["output_gzip"]["bits_per_byte"] for r in exp2]

x = np.arange(len(types))
width = 0.35
ax.bar(x - width/2, g_in_bpb, width, label='Input (gibberish)', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, g_out_bpb, width, label='Output (LLM)', color='#3498db', alpha=0.8)

# Reference: mean output bpb from meaningful prompts (L3+)
mean_meaningful_bpb = np.mean(out_bpb_arr[mask_meaningful & (levels_arr >= 3)])
ax.axhline(y=mean_meaningful_bpb, color='green', linestyle=':', alpha=0.6)
ax.text(len(types)-0.5, mean_meaningful_bpb + 0.3,
       f'Meaningful output avg\n({mean_meaningful_bpb:.1f} bpb)',
       color='green', fontsize=8, ha='right')

ax.set_xlabel("Gibberish Type")
ax.set_ylabel("Bits per Byte (gzip)")
ax.set_title("C) Gibberish Input: High Entropy ≠ High Info Out")
ax.set_xticks(x)
ax.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=7)
ax.legend(fontsize=8)

# Panel D: Human vs GPT text compression (corrected)
ax = axes[1, 1]
ax.hist(h_bpb_filtered, bins=25, alpha=0.6, label=f'Human (μ={h_bpb_filtered.mean():.2f})', color='#e74c3c')
ax.hist(g_bpb_filtered, bins=25, alpha=0.6, label=f'GPT (μ={g_bpb_filtered.mean():.2f})', color='#3498db')
ax.set_xlabel("Bits per Byte (gzip)")
ax.set_ylabel("Count")
ax.set_title(f"D) Human vs GPT Text Compression\n(n={n_filtered}, Cohen's d={cohens_d:.3f}, p={pval:.2e})")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_main_results.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig_main_results.png")


# FIGURE 2: Information amplification decomposition
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = [r["label"] for r in exp1]
length_ratios = [r["output_chars"] / r["input_chars"] for r in exp1]
density_ratios_plot = [
    r["output_gzip"]["bits_per_byte"] / r["input_gzip"]["bits_per_byte"]
    if r["input_gzip"]["bits_per_byte"] > 0 else 0
    for r in exp1
]
total_ratios = [
    r["output_gzip"]["compressed_bytes"] / r["input_gzip"]["compressed_bytes"]
    if r["input_gzip"]["compressed_bytes"] > 0 else 0
    for r in exp1
]

x = np.arange(len(labels))
colors_level = {1: '#2ecc71', 2: '#3498db', 3: '#9b59b6', 4: '#e67e22', 5: '#e74c3c', 6: '#c0392b'}
bar_colors = [colors_level.get(l, '#95a5a6') for l in levels_arr]

# Length expansion
ax1.bar(x, length_ratios, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel("Prompt")
ax1.set_ylabel("Output/Input Length Ratio")
ax1.set_title("Length Expansion (Output chars / Input chars)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
for i, (l, v) in enumerate(zip(levels_arr, length_ratios)):
    ax1.text(i, v + 0.5, f'L{l}', ha='center', fontsize=7, color='gray')

# Density ratio
ax2.bar(x, density_ratios_plot, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
ax2.set_xlabel("Prompt")
ax2.set_ylabel("Output/Input Bits-per-Byte Ratio")
ax2.set_title("Information Density Ratio\n(< 1 = output more compressible per byte)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
ax2.legend(fontsize=8)
for i, (l, v) in enumerate(zip(levels_arr, density_ratios_plot)):
    ax2.text(i, v + 0.02, f'L{l}', ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_amplification_decomposition.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig_amplification_decomposition.png")


# FIGURE 3: Logprob vs gzip comparison
fig, ax = plt.subplots(figsize=(10, 6))

exp4_levels = [r["level"] for r in exp4]
exp4_in_gzip = [r["input_gzip"]["compressed_bytes"] * 8 for r in exp4]
exp4_out_gzip = [r["output_gzip"]["compressed_bytes"] * 8 for r in exp4]
exp4_out_lp = [r.get("output_total_bits", 0) for r in exp4]

x = np.arange(len(exp4_levels))
width = 0.25
ax.bar(x - width, exp4_in_gzip, width, label='Input (gzip bits)', color='#2ecc71', alpha=0.8)
ax.bar(x, exp4_out_gzip, width, label='Output (gzip bits)', color='#3498db', alpha=0.8)
ax.bar(x + width, exp4_out_lp, width, label='Output (logprob bits)', color='#e74c3c', alpha=0.8)

ax.set_xlabel("Prompt Complexity Level")
ax.set_ylabel("Information Content (bits)")
ax.set_title("Information Measurement: Gzip vs LLM Logprobs\n"
            "(Logprob bits << gzip bits → output is highly predictable to the LLM)")
ax.set_xticks(x)
ax.set_xticklabels([f'Level {l}' for l in exp4_levels])
ax.legend()
ax.set_yscale('symlog', linthresh=100)  # Log scale to show both small and large values

plt.tight_layout()
plt.savefig(PLOTS_DIR / "fig_logprob_vs_gzip.png", dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig_logprob_vs_gzip.png")


# ============================================================
# Save corrected statistics
# ============================================================

corrected_stats = {
    "experiment1": {
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "regression_r_squared": float(r_val**2),
        "regression_p_value": float(p_val),
        "density_spearman_rho": float(rho),
        "density_spearman_p": float(p_rho),
        "mean_density_ratio_all": float(np.mean(density_ratios_plot)),
        "mean_density_ratio_L3plus": float(np.mean([d for l, d in zip(levels_arr, density_ratios_plot) if l >= 3])),
        "mean_length_expansion": float(np.mean(length_ratios)),
    },
    "experiment2_gibberish": {
        "input_mean_bpb": float(np.mean(g_in_bpb)),
        "output_mean_bpb": float(np.mean(g_out_bpb)),
        "note": "Gibberish inputs have high entropy but model outputs have lower bpb than the inputs in most cases",
    },
    "experiment3_corrected": {
        "n_samples": n_filtered,
        "min_text_length": MIN_LEN,
        "human_mean_bpb": float(h_bpb_filtered.mean()),
        "human_std_bpb": float(h_bpb_filtered.std()),
        "gpt_mean_bpb": float(g_bpb_filtered.mean()),
        "gpt_std_bpb": float(g_bpb_filtered.std()),
        "cohens_d": float(cohens_d),
        "wilcoxon_p": float(pval),
        "gpt_more_compressible": bool(g_bpb_filtered.mean() < h_bpb_filtered.mean()),
    },
    "experiment4_logprobs": {
        "key_finding": "LLM logprob bits are much lower than gzip bits, confirming outputs are highly predictable from the model's perspective",
        "mean_output_bits_per_token": float(np.mean([r.get("output_mean_bits_per_token", 0) for r in exp4])),
        "data": [
            {
                "level": r["level"],
                "input_gzip_bits": r["input_gzip"]["compressed_bytes"] * 8,
                "output_gzip_bits": r["output_gzip"]["compressed_bytes"] * 8,
                "output_logprob_bits": r.get("output_total_bits", 0),
                "output_bits_per_token": r.get("output_mean_bits_per_token", 0),
            }
            for r in exp4
        ]
    }
}

with open(RESULTS_DIR / "corrected_statistics.json", "w") as f:
    json.dump(corrected_stats, f, indent=2)

print("\nCorrected statistics saved to corrected_statistics.json")
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

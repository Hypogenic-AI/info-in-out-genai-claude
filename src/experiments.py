"""
Experiments: Does Information In ≈ Information Out in Modern GenAI?

Measures compression ratios of LLM inputs and outputs to test whether
output information content is bounded by input information content.
"""

import gzip
import json
import os
import random
import string
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from openai import OpenAI

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4.1-nano"  # Cost-efficient for many calls; upgrade for key tests
MODEL_MAIN = "gpt-4.1-mini"  # For main experiments with logprobs


def gzip_compress_ratio(text: str) -> dict:
    """Compute gzip compression statistics for a string."""
    data = text.encode("utf-8")
    compressed = gzip.compress(data, compresslevel=9)
    original_size = len(data)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    bits_per_byte = (compressed_size * 8) / original_size if original_size > 0 else 0
    return {
        "original_bytes": original_size,
        "compressed_bytes": compressed_size,
        "compression_ratio": ratio,
        "bits_per_byte": bits_per_byte,
    }


def llm_bits_per_token(text: str, model: str = MODEL_MAIN) -> dict:
    """Use LLM logprobs to estimate bits-per-token information content."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,
            logprobs=True,
            temperature=0,
        )
        # We can't get logprobs of the INPUT directly via chat API.
        # Instead, use the completions-style approach: measure how
        # compressible the text is by asking the model to continue it
        # and looking at the echo logprobs. But chat API doesn't support echo.
        # Alternative: use the model as a compressor by measuring perplexity
        # through a sliding window approach.
        return None  # Will use alternative approach
    except Exception as e:
        print(f"  logprobs error: {e}")
        return None


def estimate_llm_compressibility(text: str, model: str = MODEL_MAIN) -> dict:
    """
    Estimate how compressible text is from the LLM's perspective.

    Strategy: Ask the model to predict/complete chunks of the text.
    High prediction accuracy = high compressibility = low information.
    We use a proxy: ask the model to summarize, and compare output length
    to input length. Also use logprobs on the response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Repeat the following text exactly, word for word."},
                {"role": "user", "content": text[:500]}  # Limit for cost
            ],
            max_tokens=min(len(text.split()) + 50, 500),
            logprobs=True,
            top_logprobs=1,
            temperature=0,
        )

        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            logprobs = [t.logprob for t in response.choices[0].logprobs.content]
            # Convert log-probs (natural log) to bits
            bits = [-lp / np.log(2) for lp in logprobs]
            return {
                "mean_bits_per_token": np.mean(bits),
                "total_bits": sum(bits),
                "num_tokens": len(bits),
                "bits_per_char": sum(bits) / len(text[:500]) if text else 0,
            }
        return None
    except Exception as e:
        print(f"  LLM compressibility error: {e}")
        return None


def call_llm(prompt: str, model: str = MODEL_MAIN, max_tokens: int = 1024,
             temperature: float = 0.7, logprobs: bool = True) -> dict:
    """Call LLM and return response with metadata."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=1 if logprobs else None,
        )

        output_text = response.choices[0].message.content or ""

        result = {
            "input": prompt,
            "output": output_text,
            "model": model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        # Extract output logprobs
        if logprobs and response.choices[0].logprobs and response.choices[0].logprobs.content:
            lps = [t.logprob for t in response.choices[0].logprobs.content]
            bits = [-lp / np.log(2) for lp in lps]
            result["output_mean_bits_per_token"] = float(np.mean(bits))
            result["output_total_bits"] = float(sum(bits))
            result["output_bits_per_char"] = float(sum(bits) / len(output_text)) if output_text else 0

        return result
    except Exception as e:
        print(f"  LLM call error: {e}")
        return None


# ============================================================
# PROMPT SETS
# ============================================================

COMPLEXITY_PROMPTS = [
    # Level 1: Very simple
    {"level": 1, "label": "trivial_fact", "prompt": "What is 2+2?"},
    {"level": 1, "label": "single_word", "prompt": "Say hello."},
    {"level": 1, "label": "yes_no", "prompt": "Is the sky blue?"},

    # Level 2: Simple questions
    {"level": 2, "label": "capital", "prompt": "What is the capital of France?"},
    {"level": 2, "label": "definition", "prompt": "Define photosynthesis in one sentence."},
    {"level": 2, "label": "simple_list", "prompt": "List three primary colors."},

    # Level 3: Moderate
    {"level": 3, "label": "explain_concept", "prompt": "Explain how a refrigerator works, covering the thermodynamic cycle involved."},
    {"level": 3, "label": "compare", "prompt": "Compare and contrast TCP and UDP protocols, discussing their key differences in reliability, speed, and use cases."},
    {"level": 3, "label": "history", "prompt": "Describe the causes and consequences of the French Revolution, including economic, social, and political factors."},

    # Level 4: Complex
    {"level": 4, "label": "technical_detail", "prompt": """Explain the transformer architecture in detail, covering:
1. Self-attention mechanism and how Q, K, V matrices are computed
2. Multi-head attention and why it's beneficial
3. Position encoding approaches (sinusoidal vs learned)
4. The role of layer normalization and residual connections
5. How the encoder-decoder structure works for sequence-to-sequence tasks"""},

    {"level": 4, "label": "analysis", "prompt": """Analyze the following economic scenario and predict outcomes:
A country with 3% GDP growth rate suddenly faces:
- 15% tariff on all imports
- Central bank raises interest rates by 200 basis points
- Major trading partner enters recession
- Domestic tech sector growing at 25% annually
- Unemployment at 4.2%
- Government debt-to-GDP ratio of 85%
Discuss likely impacts on inflation, employment, trade balance, and currency value over the next 12-24 months."""},

    # Level 5: Very complex / information-dense
    {"level": 5, "label": "data_analysis", "prompt": """Given the following experimental data, perform a statistical analysis:

Group A (treatment): 23.4, 25.1, 22.8, 26.3, 24.7, 23.9, 25.5, 24.1, 26.0, 23.2,
                      25.8, 24.3, 22.5, 26.7, 24.9, 23.6, 25.2, 24.8, 26.1, 23.0
Group B (control):   20.1, 21.5, 19.8, 22.3, 20.7, 21.2, 19.5, 22.0, 20.3, 21.8,
                     19.2, 22.5, 20.9, 21.0, 19.7, 22.1, 20.5, 21.3, 19.9, 22.4

1. Calculate mean, standard deviation, and 95% CI for each group
2. Perform an independent samples t-test
3. Calculate Cohen's d effect size
4. Assess normality assumptions
5. Interpret the results in context of a clinical trial"""},

    {"level": 5, "label": "code_generation", "prompt": """Write a Python implementation of a B-tree with the following specifications:
- Order m=5 (max 4 keys per node, max 5 children)
- Support insert, search, and delete operations
- Include node splitting logic for overflow
- Include node merging/redistribution for underflow during deletion
- Add a pretty-print method that shows the tree structure
- Include comprehensive docstrings
- Add at least 5 unit tests demonstrating correctness
- Handle edge cases: empty tree, single element, duplicate keys"""},

    # Level 6: Maximum complexity
    {"level": 6, "label": "multi_domain", "prompt": """You are a consultant preparing a comprehensive report. Address ALL of the following:

SECTION 1 - TECHNICAL ARCHITECTURE:
Design a microservices architecture for a real-time trading platform that handles 100,000 orders/second. Specify: service decomposition, communication protocols (gRPC vs REST vs message queues), database choices (SQL vs NoSQL for different services), caching strategy (Redis vs Memcached), and deployment on Kubernetes with auto-scaling policies.

SECTION 2 - MATHEMATICAL MODELING:
The platform needs a risk assessment engine. Define a Value-at-Risk (VaR) model using:
- Historical simulation with 1000-day lookback
- Monte Carlo simulation with 10,000 paths
- Parametric VaR assuming normal and t-distributed returns
Compare the three approaches mathematically and discuss when each is most appropriate.

SECTION 3 - REGULATORY COMPLIANCE:
Outline compliance requirements for: MiFID II (EU), Dodd-Frank (US), and MAS regulations (Singapore). How does the architecture need to adapt for: trade reporting, best execution obligations, position limits, and data residency requirements?

SECTION 4 - PERFORMANCE OPTIMIZATION:
The matching engine must achieve <50μs latency. Discuss: kernel bypass networking (DPDK/RDMA), lock-free data structures for the order book, CPU cache optimization, NUMA-aware memory allocation, and the trade-offs of using FPGA vs software for matching.

Provide concrete numbers, formulas, and code snippets where appropriate."""},
]

GIBBERISH_PROMPTS = [
    {"type": "random_chars", "prompt": "".join(random.choices(string.ascii_letters + string.digits, k=100))},
    {"type": "random_chars_long", "prompt": "".join(random.choices(string.ascii_letters + string.digits, k=500))},
    {"type": "shuffled_words", "prompt": "banana quantum if the 73 protocol xyz running marbles abstract ceiling telephone"},
    {"type": "keyboard_mash", "prompt": "asdfghjkl qwerty zxcvbnm poiuytrewq lkjhgfdsa mnbvcxz"},
    {"type": "repeated_nonsense", "prompt": "blorp fleem gnarx zibbit wompus " * 10},
    {"type": "unicode_noise", "prompt": "αβγδ ∂∆∏∑ ℜℑ∀∃ ⊕⊗⊘⊙ ≤≥≠≈ ∞∇∫∮ ♠♣♥♦ ★☆○●"},
]


def run_experiment_1():
    """Experiment 1: Input complexity vs output complexity."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Input Complexity vs Output Complexity")
    print("="*60)

    results = []
    for i, item in enumerate(COMPLEXITY_PROMPTS):
        print(f"\n  [{i+1}/{len(COMPLEXITY_PROMPTS)}] Level {item['level']}: {item['label']}")

        # Call LLM
        resp = call_llm(item["prompt"], model=MODEL_MAIN, max_tokens=2048, temperature=0.7)
        if resp is None:
            continue

        # Measure compression of input and output
        input_comp = gzip_compress_ratio(item["prompt"])
        output_comp = gzip_compress_ratio(resp["output"])

        result = {
            "level": item["level"],
            "label": item["label"],
            "input_text": item["prompt"],
            "output_text": resp["output"],
            "input_chars": len(item["prompt"]),
            "output_chars": len(resp["output"]),
            "input_tokens": resp["input_tokens"],
            "output_tokens": resp["output_tokens"],
            "input_gzip": input_comp,
            "output_gzip": output_comp,
            "output_mean_bits_per_token": resp.get("output_mean_bits_per_token"),
            "output_total_bits": resp.get("output_total_bits"),
            "output_bits_per_char": resp.get("output_bits_per_char"),
        }
        results.append(result)

        print(f"    Input: {input_comp['original_bytes']}B -> {input_comp['compressed_bytes']}B "
              f"(ratio={input_comp['compression_ratio']:.2f}, bpb={input_comp['bits_per_byte']:.2f})")
        print(f"    Output: {output_comp['original_bytes']}B -> {output_comp['compressed_bytes']}B "
              f"(ratio={output_comp['compression_ratio']:.2f}, bpb={output_comp['bits_per_byte']:.2f})")
        if resp.get("output_mean_bits_per_token"):
            print(f"    Output LLM bits/token: {resp['output_mean_bits_per_token']:.2f}")

        time.sleep(0.5)  # Rate limiting

    # Save results
    with open(RESULTS_DIR / "experiment1_complexity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_experiment_2():
    """Experiment 2: Gibberish/random input control."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Gibberish Control")
    print("="*60)

    results = []
    for i, item in enumerate(GIBBERISH_PROMPTS):
        print(f"\n  [{i+1}/{len(GIBBERISH_PROMPTS)}] {item['type']}")

        resp = call_llm(item["prompt"], model=MODEL_MAIN, max_tokens=512, temperature=0.7)
        if resp is None:
            continue

        input_comp = gzip_compress_ratio(item["prompt"])
        output_comp = gzip_compress_ratio(resp["output"])

        result = {
            "type": item["type"],
            "input_text": item["prompt"],
            "output_text": resp["output"],
            "input_chars": len(item["prompt"]),
            "output_chars": len(resp["output"]),
            "input_gzip": input_comp,
            "output_gzip": output_comp,
            "output_mean_bits_per_token": resp.get("output_mean_bits_per_token"),
            "output_total_bits": resp.get("output_total_bits"),
        }
        results.append(result)

        print(f"    Input: {input_comp['original_bytes']}B (bpb={input_comp['bits_per_byte']:.2f})")
        print(f"    Output: {output_comp['original_bytes']}B (bpb={output_comp['bits_per_byte']:.2f})")

        time.sleep(0.5)

    with open(RESULTS_DIR / "experiment2_gibberish.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_experiment_3():
    """Experiment 3: Human vs LLM text compression (GPT-Wiki-Intro dataset)."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Human vs LLM Text Compression")
    print("="*60)

    # Load dataset
    human_path = DATASETS_DIR / "gpt_wiki_intro" / "human_texts.txt"
    gpt_path = DATASETS_DIR / "gpt_wiki_intro" / "gpt_texts.txt"

    if not human_path.exists() or not gpt_path.exists():
        print("  Dataset not found, skipping...")
        return []

    with open(human_path) as f:
        human_texts = f.read().strip().split("\n")
    with open(gpt_path) as f:
        gpt_texts = f.read().strip().split("\n")

    # Use first 200 samples for speed
    n_samples = min(200, len(human_texts), len(gpt_texts))
    human_texts = human_texts[:n_samples]
    gpt_texts = gpt_texts[:n_samples]

    print(f"  Loaded {n_samples} paired samples")

    results = {"human": [], "gpt": []}

    for i in range(n_samples):
        h_comp = gzip_compress_ratio(human_texts[i])
        g_comp = gzip_compress_ratio(gpt_texts[i])
        results["human"].append(h_comp)
        results["gpt"].append(g_comp)

    # Summary stats
    h_ratios = [r["compression_ratio"] for r in results["human"]]
    g_ratios = [r["compression_ratio"] for r in results["gpt"]]
    h_bpb = [r["bits_per_byte"] for r in results["human"]]
    g_bpb = [r["bits_per_byte"] for r in results["gpt"]]

    print(f"\n  Human text: mean ratio={np.mean(h_ratios):.3f} ± {np.std(h_ratios):.3f}, "
          f"mean bpb={np.mean(h_bpb):.3f}")
    print(f"  GPT text:   mean ratio={np.mean(g_ratios):.3f} ± {np.std(g_ratios):.3f}, "
          f"mean bpb={np.mean(g_bpb):.3f}")

    # Statistical test
    stat, pval = stats.wilcoxon(h_ratios, g_ratios)
    print(f"  Wilcoxon test: stat={stat:.1f}, p={pval:.2e}")

    with open(RESULTS_DIR / "experiment3_human_vs_gpt.json", "w") as f:
        json.dump({
            "n_samples": n_samples,
            "human_ratios": h_ratios,
            "gpt_ratios": g_ratios,
            "human_bpb": h_bpb,
            "gpt_bpb": g_bpb,
            "human_mean_ratio": float(np.mean(h_ratios)),
            "gpt_mean_ratio": float(np.mean(g_ratios)),
            "human_mean_bpb": float(np.mean(h_bpb)),
            "gpt_mean_bpb": float(np.mean(g_bpb)),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(pval),
        }, f, indent=2)

    return results


def run_experiment_4():
    """Experiment 4: LLM logprob-based information measurement.

    Measure output information using logprobs from the generating model.
    Compare against gzip-based measurement.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: LLM Logprob Information Measurement")
    print("="*60)

    # Select a subset of prompts at different complexity levels
    test_prompts = [
        {"level": 1, "prompt": "What color is grass?"},
        {"level": 2, "prompt": "Explain why the sky appears blue."},
        {"level": 3, "prompt": "Describe the process of nuclear fusion in stars, including the proton-proton chain and the CNO cycle."},
        {"level": 4, "prompt": "Compare the computational complexity of sorting algorithms (merge sort, quicksort, heapsort, timsort) "
                               "in terms of worst-case, average-case, and best-case time complexity, space complexity, "
                               "stability, and practical performance on modern hardware with cache effects."},
        {"level": 5, "prompt": """Analyze this dataset and provide insights:
Year,Revenue($M),Employees,R&D_Spend($M),Market_Share(%),Customer_Satisfaction(1-10)
2019,45.2,120,8.3,12.1,7.2
2020,38.7,115,9.1,11.8,6.9
2021,52.1,135,11.2,13.5,7.5
2022,67.8,180,14.5,15.2,7.8
2023,71.3,195,16.2,16.1,8.1
2024,82.5,220,19.8,17.3,8.4

Provide: trend analysis, correlation analysis between variables, growth rate calculations,
forecasting for 2025-2026, and strategic recommendations based on the data patterns."""},
    ]

    results = []
    for i, item in enumerate(test_prompts):
        print(f"\n  [{i+1}/{len(test_prompts)}] Level {item['level']}")

        resp = call_llm(item["prompt"], model=MODEL_MAIN, max_tokens=2048,
                        temperature=0.7, logprobs=True)
        if resp is None:
            continue

        input_comp = gzip_compress_ratio(item["prompt"])
        output_comp = gzip_compress_ratio(resp["output"])

        result = {
            "level": item["level"],
            "input_text": item["prompt"],
            "output_text": resp["output"],
            "input_chars": len(item["prompt"]),
            "output_chars": len(resp["output"]),
            "input_gzip": input_comp,
            "output_gzip": output_comp,
            "output_mean_bits_per_token": resp.get("output_mean_bits_per_token"),
            "output_total_bits": resp.get("output_total_bits"),
            "output_bits_per_char": resp.get("output_bits_per_char"),
            # Information amplification
            "gzip_info_ratio": (output_comp["compressed_bytes"] * 8) / (input_comp["compressed_bytes"] * 8) if input_comp["compressed_bytes"] > 0 else None,
        }
        results.append(result)

        print(f"    Input: {input_comp['compressed_bytes']*8} bits (gzip)")
        print(f"    Output: {output_comp['compressed_bytes']*8} bits (gzip)")
        if resp.get("output_total_bits"):
            print(f"    Output: {resp['output_total_bits']:.0f} bits (logprob)")
            print(f"    Output bits/token: {resp['output_mean_bits_per_token']:.2f}")
        print(f"    Info amplification (gzip): {result['gzip_info_ratio']:.2f}x" if result['gzip_info_ratio'] else "")

        time.sleep(0.5)

    with open(RESULTS_DIR / "experiment4_logprobs.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ============================================================
# ANALYSIS AND VISUALIZATION
# ============================================================

def analyze_and_plot(exp1_results, exp2_results, exp3_results, exp4_results):
    """Create all analysis plots and compute statistics."""
    print("\n" + "="*60)
    print("ANALYSIS AND VISUALIZATION")
    print("="*60)

    fig_num = 0
    stats_summary = {}

    # --- PLOT 1: Input vs Output complexity (gzip bits) ---
    if exp1_results:
        fig_num += 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        levels = [r["level"] for r in exp1_results]
        input_bits = [r["input_gzip"]["compressed_bytes"] * 8 for r in exp1_results]
        output_bits = [r["output_gzip"]["compressed_bytes"] * 8 for r in exp1_results]
        input_bpb = [r["input_gzip"]["bits_per_byte"] for r in exp1_results]
        output_bpb = [r["output_gzip"]["bits_per_byte"] for r in exp1_results]

        # Panel A: Total bits
        colors = plt.cm.viridis(np.array(levels) / max(levels))
        ax1.scatter(input_bits, output_bits, c=levels, cmap='viridis', s=100,
                   edgecolors='black', linewidth=0.5, zorder=5)

        # Add identity line and regression
        max_val = max(max(input_bits), max(output_bits))
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y = x (identity)')

        # Regression
        slope, intercept, r_val, p_val, std_err = stats.linregress(input_bits, output_bits)
        x_fit = np.linspace(min(input_bits), max(input_bits), 100)
        ax1.plot(x_fit, slope * x_fit + intercept, 'r-', alpha=0.7,
                label=f'Fit: y={slope:.1f}x+{intercept:.0f}\nr²={r_val**2:.3f}, p={p_val:.2e}')

        ax1.set_xlabel("Input Information (gzip compressed bits)", fontsize=12)
        ax1.set_ylabel("Output Information (gzip compressed bits)", fontsize=12)
        ax1.set_title("A) Input vs Output Information Content", fontsize=13)
        ax1.legend(fontsize=9)
        cbar = plt.colorbar(ax1.scatter(input_bits, output_bits, c=levels, cmap='viridis', s=0), ax=ax1)
        cbar.set_label("Prompt Complexity Level")

        stats_summary["exp1_total_bits"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val**2),
            "p_value": float(p_val),
            "correlation": float(r_val),
        }

        # Panel B: Bits per byte (information density)
        ax2.scatter(input_bpb, output_bpb, c=levels, cmap='viridis', s=100,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax2.axhline(y=np.mean(output_bpb), color='r', linestyle=':', alpha=0.5,
                   label=f'Output mean={np.mean(output_bpb):.2f}')
        ax2.axvline(x=np.mean(input_bpb), color='b', linestyle=':', alpha=0.5,
                   label=f'Input mean={np.mean(input_bpb):.2f}')

        ax2.set_xlabel("Input Bits per Byte (gzip)", fontsize=12)
        ax2.set_ylabel("Output Bits per Byte (gzip)", fontsize=12)
        ax2.set_title("B) Information Density: Input vs Output", fontsize=13)
        ax2.legend(fontsize=9)
        cbar2 = plt.colorbar(ax2.scatter(input_bpb, output_bpb, c=levels, cmap='viridis', s=0), ax=ax2)
        cbar2.set_label("Prompt Complexity Level")

        # Spearman correlation for density
        rho, p_rho = stats.spearmanr(input_bpb, output_bpb)
        stats_summary["exp1_density"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "input_mean_bpb": float(np.mean(input_bpb)),
            "output_mean_bpb": float(np.mean(output_bpb)),
        }

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig1_input_vs_output_complexity.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig1_input_vs_output_complexity.png")
        print(f"    Regression: slope={slope:.2f}, r²={r_val**2:.3f}, p={p_val:.2e}")
        print(f"    Spearman (density): rho={rho:.3f}, p={p_rho:.3e}")

    # --- PLOT 2: Gibberish control ---
    if exp2_results:
        fig_num += 1
        fig, ax = plt.subplots(figsize=(10, 6))

        types = [r["type"] for r in exp2_results]
        input_bpb = [r["input_gzip"]["bits_per_byte"] for r in exp2_results]
        output_bpb = [r["output_gzip"]["bits_per_byte"] for r in exp2_results]

        x = np.arange(len(types))
        width = 0.35
        ax.bar(x - width/2, input_bpb, width, label='Input (gibberish)', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, output_bpb, width, label='Output (LLM response)', color='#3498db', alpha=0.8)

        ax.set_xlabel("Gibberish Type", fontsize=12)
        ax.set_ylabel("Bits per Byte (gzip)", fontsize=12)
        ax.set_title("Gibberish Input: High Entropy In ≠ High Information Out", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.axhline(y=8, color='gray', linestyle='--', alpha=0.3, label='Max (8 bpb)')

        # Add meaningful vs gibberish comparison
        # Get mean output bpb from experiment 1 for reference
        if exp1_results:
            exp1_output_bpb_mean = np.mean([r["output_gzip"]["bits_per_byte"] for r in exp1_results])
            ax.axhline(y=exp1_output_bpb_mean, color='green', linestyle=':', alpha=0.5)
            ax.text(len(types)-1, exp1_output_bpb_mean + 0.1, f'Meaningful output avg ({exp1_output_bpb_mean:.2f})',
                   color='green', fontsize=8, ha='right')

        stats_summary["exp2_gibberish"] = {
            "input_mean_bpb": float(np.mean(input_bpb)),
            "output_mean_bpb": float(np.mean(output_bpb)),
            "types": types,
            "input_bpb": [float(x) for x in input_bpb],
            "output_bpb": [float(x) for x in output_bpb],
        }

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig2_gibberish_control.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig2_gibberish_control.png")

    # --- PLOT 3: Human vs GPT text compression ---
    if exp3_results:
        fig_num += 1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        h_ratios = [r["compression_ratio"] for r in exp3_results["human"]]
        g_ratios = [r["compression_ratio"] for r in exp3_results["gpt"]]
        h_bpb = [r["bits_per_byte"] for r in exp3_results["human"]]
        g_bpb = [r["bits_per_byte"] for r in exp3_results["gpt"]]

        # Panel A: Distribution comparison
        ax1.hist(h_bpb, bins=30, alpha=0.6, label=f'Human (μ={np.mean(h_bpb):.3f})', color='#e74c3c')
        ax1.hist(g_bpb, bins=30, alpha=0.6, label=f'GPT (μ={np.mean(g_bpb):.3f})', color='#3498db')
        ax1.set_xlabel("Bits per Byte (gzip)", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("A) Human vs GPT Text: Compression Distribution", fontsize=13)
        ax1.legend(fontsize=10)

        # Panel B: Paired comparison
        ax2.scatter(h_bpb, g_bpb, alpha=0.4, s=30, color='purple')
        ax2.plot([min(h_bpb + g_bpb), max(h_bpb + g_bpb)],
                [min(h_bpb + g_bpb), max(h_bpb + g_bpb)],
                'k--', alpha=0.3, label='y = x')
        ax2.set_xlabel("Human Text Bits per Byte", fontsize=12)
        ax2.set_ylabel("GPT Text Bits per Byte", fontsize=12)
        ax2.set_title("B) Paired Comparison", fontsize=13)
        ax2.legend()

        # Effect size
        cohens_d = (np.mean(h_bpb) - np.mean(g_bpb)) / np.sqrt((np.std(h_bpb)**2 + np.std(g_bpb)**2) / 2)
        stat, pval = stats.wilcoxon(h_bpb, g_bpb)

        stats_summary["exp3_human_vs_gpt"] = {
            "human_mean_bpb": float(np.mean(h_bpb)),
            "gpt_mean_bpb": float(np.mean(g_bpb)),
            "cohens_d": float(cohens_d),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(pval),
            "n_samples": len(h_bpb),
        }

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig3_human_vs_gpt_compression.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig3_human_vs_gpt_compression.png")
        print(f"    Cohen's d: {cohens_d:.3f}, Wilcoxon p={pval:.2e}")

    # --- PLOT 4: Information amplification factor ---
    if exp1_results:
        fig_num += 1
        fig, ax = plt.subplots(figsize=(10, 6))

        levels = [r["level"] for r in exp1_results]
        labels = [r["label"] for r in exp1_results]
        amp_factors = [
            (r["output_gzip"]["compressed_bytes"]) / (r["input_gzip"]["compressed_bytes"])
            if r["input_gzip"]["compressed_bytes"] > 0 else 0
            for r in exp1_results
        ]

        colors_map = {1: '#2ecc71', 2: '#3498db', 3: '#9b59b6', 4: '#e67e22', 5: '#e74c3c', 6: '#c0392b'}
        bar_colors = [colors_map.get(l, '#95a5a6') for l in levels]

        x = np.arange(len(labels))
        bars = ax.bar(x, amp_factors, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Amplification = 1x (break-even)')
        ax.set_xlabel("Prompt", fontsize=12)
        ax.set_ylabel("Output/Input Information Ratio (gzip bits)", fontsize=12)
        ax.set_title("Information Amplification Factor by Prompt Complexity", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend()

        # Add level annotations
        for i, (l, a) in enumerate(zip(levels, amp_factors)):
            ax.text(i, a + 0.1, f'L{l}', ha='center', fontsize=7, color='gray')

        stats_summary["exp1_amplification"] = {
            "mean_amplification": float(np.mean(amp_factors)),
            "min_amplification": float(np.min(amp_factors)),
            "max_amplification": float(np.max(amp_factors)),
            "amplification_by_prompt": {l: float(a) for l, a in zip(labels, amp_factors)},
        }

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig4_information_amplification.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig4_information_amplification.png")
        print(f"    Mean amplification: {np.mean(amp_factors):.2f}x")

    # --- PLOT 5: LLM logprob-based measurement ---
    if exp4_results:
        fig_num += 1
        fig, ax = plt.subplots(figsize=(10, 6))

        levels = [r["level"] for r in exp4_results]
        input_gzip_bits = [r["input_gzip"]["compressed_bytes"] * 8 for r in exp4_results]
        output_logprob_bits = [r.get("output_total_bits", 0) for r in exp4_results if r.get("output_total_bits")]
        output_gzip_bits = [r["output_gzip"]["compressed_bytes"] * 8 for r in exp4_results]

        ax.scatter(input_gzip_bits, output_gzip_bits, s=100, label='Output (gzip bits)',
                  marker='o', color='#3498db', edgecolors='black', zorder=5)

        if output_logprob_bits and len(output_logprob_bits) == len(input_gzip_bits):
            ax.scatter(input_gzip_bits, output_logprob_bits, s=100, label='Output (logprob bits)',
                      marker='^', color='#e74c3c', edgecolors='black', zorder=5)

        max_val = max(max(input_gzip_bits), max(output_gzip_bits))
        ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3, label='y = x')

        for i, l in enumerate(levels):
            ax.annotate(f'L{l}', (input_gzip_bits[i], output_gzip_bits[i]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel("Input Information (gzip compressed bits)", fontsize=12)
        ax.set_ylabel("Output Information (bits)", fontsize=12)
        ax.set_title("Input vs Output Information: Gzip and Logprob Measures", fontsize=13)
        ax.legend()

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig5_logprob_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig5_logprob_comparison.png")

    # --- PLOT 6: Summary / Key finding ---
    if exp1_results and exp2_results:
        fig_num += 1
        fig, ax = plt.subplots(figsize=(12, 7))

        # Meaningful prompts
        m_input = [r["input_gzip"]["compressed_bytes"] * 8 for r in exp1_results]
        m_output = [r["output_gzip"]["compressed_bytes"] * 8 for r in exp1_results]
        m_levels = [r["level"] for r in exp1_results]

        # Gibberish prompts
        g_input = [r["input_gzip"]["compressed_bytes"] * 8 for r in exp2_results]
        g_output = [r["output_gzip"]["compressed_bytes"] * 8 for r in exp2_results]

        scatter_m = ax.scatter(m_input, m_output, c=m_levels, cmap='viridis', s=120,
                              edgecolors='black', linewidth=0.5, zorder=5, label='Meaningful prompts')
        ax.scatter(g_input, g_output, c='red', s=120, marker='x', linewidth=2,
                  zorder=5, label='Gibberish prompts')

        # Regression for meaningful
        slope, intercept, r_val, p_val, _ = stats.linregress(m_input, m_output)
        x_fit = np.linspace(0, max(m_input) * 1.1, 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'b-', alpha=0.5,
               label=f'Meaningful fit: y={slope:.1f}x+{intercept:.0f} (r²={r_val**2:.3f})')

        # Identity line
        max_val = max(max(m_input + g_input), max(m_output + g_output))
        ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3, label='y = x (identity)')

        ax.set_xlabel("Input Information (gzip compressed bits)", fontsize=13)
        ax.set_ylabel("Output Information (gzip compressed bits)", fontsize=13)
        ax.set_title("Information In vs Information Out in LLM Generation\n"
                    "(Meaningful prompts show positive correlation; gibberish does not)", fontsize=14)
        ax.legend(loc='upper left', fontsize=10)

        cbar = plt.colorbar(scatter_m, ax=ax)
        cbar.set_label("Prompt Complexity Level", fontsize=11)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "fig6_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig6_summary.png")

    # Save all statistics
    with open(RESULTS_DIR / "statistics_summary.json", "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\n  Statistics saved to statistics_summary.json")

    return stats_summary


def main():
    """Run all experiments and analysis."""
    print("="*60)
    print("RESEARCH: Does Information In ≈ Information Out in GenAI?")
    print("="*60)
    print(f"Model (main): {MODEL_MAIN}")
    print(f"Seed: {SEED}")

    # Run experiments
    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    exp3 = run_experiment_3()
    exp4 = run_experiment_4()

    # Analysis and visualization
    stats_summary = analyze_and_plot(exp1, exp2, exp3, exp4)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)

    return stats_summary


if __name__ == "__main__":
    main()

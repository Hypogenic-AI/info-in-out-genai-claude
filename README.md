# Does Information In ≈ Information Out in Modern GenAI?

Empirical investigation of whether generative AI output information content is bounded by input information content, measured via compression ratios.

## Key Findings

- **Output information density is consistently lower than input density**: For complex prompts, LLM outputs have ~0.49× the bits-per-byte of inputs (gzip). The model acts as a "decompressor," expanding compressed prompts into longer, more predictable text.
- **Total output bits exceed input bits due to length expansion** (6-44×), not higher information density. The "extra" information comes from model weights.
- **By the model's own logprobs, outputs contain ~0.3 bits/token** of surprise — nearly perfectly predictable from the model's perspective.
- **Gibberish inputs don't produce high-information outputs**: High raw entropy ≠ usable information for aligned LLMs.
- **Compressor choice matters**: GPT text is *less* compressible by gzip but *more* compressible by LLMs (vs human text).

## Project Structure

```
├── REPORT.md              # Full research report with results
├── planning.md            # Experimental design and methodology
├── src/
│   ├── experiments.py     # Main experiment code (4 experiments)
│   └── analysis.py        # Corrected analysis and final figures
├── results/
│   ├── plots/             # All figures (PNG)
│   ├── experiment1_complexity.json
│   ├── experiment2_gibberish.json
│   ├── experiment3_human_vs_gpt.json
│   ├── experiment4_logprobs.json
│   ├── statistics_summary.json
│   └── corrected_statistics.json
├── datasets/              # Pre-downloaded datasets
├── papers/                # Reference papers (PDFs)
├── code/                  # Cloned baseline repositories
└── literature_review.md   # Literature synthesis
```

## Reproducing Results

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy matplotlib scipy pandas tqdm

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiments
python src/experiments.py

# Run corrected analysis
python src/analysis.py
```

Requires: Python 3.10+, OpenAI API key (GPT-4.1-mini). No GPU needed.

## Method

1. Send prompts of varying complexity (6 levels, 14 prompts) to GPT-4.1-mini
2. Measure gzip compression ratio of both input and output
3. Measure LLM logprob-based information content of outputs
4. Control: test gibberish/random inputs
5. Validate: compare human vs GPT text compression on GPT-Wiki-Intro dataset

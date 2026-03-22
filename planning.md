# Research Plan: Does Information In ≈ Information Out in Modern GenAI?

## Motivation & Novelty Assessment

### Why This Research Matters
Generative AI systems appear to produce rich, complex outputs from simple prompts — a user types a sentence and gets paragraphs of text or detailed images. This raises a fundamental information-theoretic question: where does the "extra" information come from? Understanding whether GenAI outputs are informationally bounded by their inputs has implications for AI safety (can models leak training data?), copyright (do outputs contain memorized content?), and fundamental understanding of what these models actually do.

### Gap in Existing Work
The literature establishes that language modeling IS compression (Delétang et al., 2024) and that LLM-generated text is far more compressible than human text when measured against an LLM prior (Wang et al., 2025). However, **no existing work systematically measures both input and output information content simultaneously and plots their relationship**. Papers study output compression in isolation. The ACR paper (Schwarzschild et al., 2024) begins to address input-output ratios but only for memorized content. Our contribution fills this gap by directly measuring and plotting input complexity vs. output complexity across varying input types.

### Our Novel Contribution
We provide the first systematic empirical measurement of the input-output information relationship in GenAI by:
1. Measuring compression ratios of both prompts and responses using multiple compressors (gzip, LLM-based log-probability)
2. Varying input complexity systematically to trace the relationship curve
3. Testing across text (LLM) and image (text-to-image) modalities
4. Distinguishing meaningful information from raw entropy (gibberish inputs)

### Experiment Justification
- **Experiment 1 (LLM text, varying complexity)**: Core test — do LLM outputs track input complexity? Uses real API calls with prompts of increasing information density.
- **Experiment 2 (Gibberish control)**: Tests the caveat that raw entropy ≠ usable information. Models should NOT produce complex outputs from gibberish.
- **Experiment 3 (GPT-Wiki-Intro dataset)**: Uses existing paired human/LLM text to validate that LLM outputs are more compressible (less information) than human text.
- **Experiment 4 (LLM-based compression measurement)**: Uses log-probabilities from a reference LLM to measure information content more precisely than gzip.

## Research Question
Does the information content of a generative model's output track (and remain bounded by) the information content of its input, when information is measured via compression ratio?

## Hypothesis Decomposition
1. **H1**: Output compression ratio correlates positively with input compression ratio — more complex inputs yield more complex outputs.
2. **H2**: Output information content (bits) does not systematically exceed input information content when measured against the model's own prior.
3. **H3**: Gibberish/random inputs produce low-information outputs despite having high raw entropy — the model routes only meaningful information.
4. **H4**: LLM-generated text is more compressible than human text (replication of Wang et al., 2025).

## Proposed Methodology

### Approach
We use two complementary measures of information content:
1. **Gzip compression ratio**: Fast, universal, captures syntactic/statistical complexity
2. **LLM log-probability (bits-per-token)**: Uses a reference model's perplexity as a compression measure — directly measures how surprising the text is to an LLM

For each input-output pair from an LLM, we compute both measures on the input (prompt) and output (response), then plot the relationship.

### Experimental Steps

#### Experiment 1: Input Complexity vs Output Complexity
1. Create prompts spanning a complexity spectrum:
   - Simple factual questions ("What is the capital of France?")
   - Medium complexity (multi-step reasoning, technical explanations)
   - High complexity (detailed technical prompts with constraints)
   - Very high complexity (long, information-dense prompts with data)
2. Send each prompt to GPT-4.1 via API
3. Measure gzip compression ratio of input and output
4. Measure LLM-based bits-per-token of input and output (using GPT-4.1 logprobs)
5. Plot input complexity vs output complexity

#### Experiment 2: Gibberish Control
1. Generate random character strings, shuffled words, and nonsense
2. Send to GPT-4.1
3. Measure compression ratios — expect high input entropy but low output information

#### Experiment 3: Human vs LLM Text Compression
1. Load GPT-Wiki-Intro dataset (paired human/GPT texts)
2. Compute gzip compression ratios for both
3. Compare distributions

#### Experiment 4: LLM Log-Prob Measurement
1. Use GPT-4.1 with logprobs enabled
2. For each output token, record log-probability
3. Sum negative log-probs = bits of information (cross-entropy)
4. Compare input bits vs output bits

### Baselines
- Gzip compression ratio as baseline information measure
- Random text compression ratio as upper bound on entropy
- Human-written text compression ratio as reference

### Evaluation Metrics
- **Compression ratio** (R = uncompressed_size / compressed_size)
- **Bits per byte** (compressed_bits / original_bytes)
- **Mean log-probability** (from LLM logprobs)
- **Pearson/Spearman correlation** between input and output complexity
- **Information amplification factor** (output_bits / input_bits)

### Statistical Analysis Plan
- Pearson and Spearman correlation for input-output complexity relationship
- Paired t-test or Wilcoxon signed-rank for human vs LLM text compression
- Bootstrap confidence intervals for correlation estimates
- Significance level: α = 0.05

## Expected Outcomes
- **Supporting hypothesis**: Output complexity tracks input complexity with slope ≤ 1. Gibberish inputs produce low-complexity outputs. LLM text more compressible than human text.
- **Refuting hypothesis**: Output complexity exceeds input complexity systematically, even accounting for model prior. The model generates genuinely novel information not traceable to input or training.

## Timeline and Milestones
1. Planning: 15 min (this document)
2. Environment setup: 10 min
3. Implementation: 60 min
4. Experiments: 60 min
5. Analysis & visualization: 30 min
6. Documentation: 20 min

## Potential Challenges
- API rate limits or costs — mitigate with batching and caching
- Gzip may not capture semantic complexity well — supplement with LLM logprobs
- Short texts compress poorly with gzip — use concatenation or minimum length thresholds
- Model refusal on gibberish — document as supporting evidence for H3

## Success Criteria
- Clear plot showing input vs output complexity relationship
- Statistical test results for correlation significance
- Gibberish control showing expected behavior
- Replication of LLM text compressibility finding

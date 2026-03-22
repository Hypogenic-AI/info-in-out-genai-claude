# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: "Does information in ≈ information out in modern GenAI?" The research investigates whether generative model output information content is bounded by input information content, using compression ratio as the primary measurement.

## Papers

Total papers downloaded: 16

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Language Modeling Is Compression | Delétang et al. | 2024 | papers/deletang2024_*.pdf | Foundational: LM = compression |
| Lossless Compression of LLM-Generated Text | Wang et al. | 2025 | papers/wang2025_*.pdf | LLM text 20x compressible |
| LLMZip | Valmeekam et al. | 2023 | papers/valmeekam2023_*.pdf | English entropy ~0.71 bpc |
| Adversarial Compression Ratio | Schwarzschild et al. | 2024 | papers/schwarzschild2024_*.pdf | ACR metric for memorization |
| Ranking LLMs by Compression | Guo et al. | 2024 | papers/huang2024_*.pdf | Compression ratio as LLM metric |
| Compression Laws for LLMs | Huang et al. | 2025 | papers/huang2025_*.pdf | Scaling laws for compression |
| Gzip Text Classification | Jiang et al. | 2023 | papers/jiang2023_*.pdf | NCD metric, gzip for classification |
| Neural NCD Disconnect | Bao et al. | 2024 | papers/bao2024_*.pdf | Compression ≠ classification |
| Semantic Uncertainty | Kuhn et al. | 2023 | papers/kuhn2023_*.pdf | Semantic entropy for LLMs |
| Locally Typical Sampling | Meister et al. | 2023 | papers/meister2023_*.pdf | Information-theoretic generation |
| Generative Diversity IB | Chen et al. | 2025 | papers/chen2025_*.pdf | Information bottleneck in generation |
| Entropy to Epiplexity | Li et al. | 2026 | papers/li2026_*.pdf | Info for bounded observers |
| Top-H Decoding | Wu et al. | 2025 | papers/wu2025_*.pdf | Bounded entropy generation |
| 500xCompressor | Li et al. | 2024 | papers/li2024_500x*.pdf | Extreme prompt compression |
| Adversarial Diffusion Compression | Li et al. | 2024 | papers/li2024_adversarial_*.pdf | Image compression |
| Locally Typical (earlier) | Meister et al. | 2022 | papers/meister2022_*.pdf | Earlier version |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets downloaded: 5

| Name | Source | Size | Purpose | Location | Notes |
|------|--------|------|---------|----------|-------|
| enwik8 | mattmahoney.net | 96 MB | Standard text compression benchmark | datasets/enwik8/ | Wikipedia XML, first 10^8 bytes |
| text8 | mattmahoney.net | 96 MB | Text compression benchmark | datasets/text8/ | Lowercase Wikipedia |
| GPT-Wiki-Intro | HuggingFace | 15 MB | Human vs LLM text comparison | datasets/gpt_wiki_intro/ | 5K paired examples |
| AI-Human Text | HuggingFace | 17 MB | Human vs AI text classification | datasets/ai_human_text/ | 10K examples |
| Sample Images | Generated | <1 MB | Cross-modal compression test | datasets/sample_images/ | 5 test images |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| language_modeling_is_compression | github.com/google-deepmind/... | Core paper implementation | code/language_modeling_is_compression/ | Arithmetic coding + LLM |
| LLMzip | github.com/... | LLaMA-based text compression | code/LLMzip/ | Original LLMZip implementation |
| llm-compression-intelligence | github.com/HKUST-NLP/... | Compression = intelligence | code/llm-compression-intelligence/ | Links benchmarks to BPC |
| FineZip | github.com/... | Fast LLM compression | code/FineZip/ | 54x faster than LLMzip |
| llama-zip | github.com/... | Production LLM compression | code/llama-zip/ | Best empirical benchmarks |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for initial discovery
2. Web search on Google/arXiv for specific topics: "LLM compression", "information content generative AI", "compression ratio input output"
3. Followed citation chains from core papers
4. Searched GitHub for implementations

### Selection Criteria
- Papers directly addressing compression ↔ language modeling equivalence (highest priority)
- Papers measuring information content of LLM outputs vs inputs
- Papers providing practical compression-ratio measurement methodologies
- Papers with available code for reproducibility
- Papers covering both text and image modalities

### Challenges Encountered
- Few papers directly address the "information in vs out" question; most focus on compression efficiency of LLM outputs in isolation
- Image generation information content is understudied — no direct papers measuring text prompt info vs generated image info
- The "adjusted compression rate" (accounting for model parameters) changes the picture dramatically but is rarely discussed

### Gaps and Workarounds
- **No image generation papers found**: Will need to design experiments measuring text prompt compression vs generated image compression from scratch
- **No formal input-output information framework**: Will use compression ratio comparison (compress input, compress output, compare) as our operational framework
- **Limited access to large LLMs**: Experiments should use API-accessible models or small open models

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **GPT-Wiki-Intro**: Paired human/GPT text for direct comparison (already downloaded)
- **AI-Human-Generated-Text**: Additional human vs AI comparison
- **Custom generated data**: Generate outputs from LLMs with controlled input complexity

### 2. Baseline Methods
- **gzip**: Fast, universal, well-understood compression
- **zlib**: Similar to gzip, good for programmatic use
- **LZMA**: Higher compression ratio baseline
- Compare compression ratios of: (a) input prompts alone, (b) outputs alone, (c) input+output concatenated

### 3. Evaluation Metrics
- **Compression ratio**: R = original_size / compressed_size
- **Information amplification factor**: R_output / R_input
- **Bits per byte**: Normalized information content
- **Conditional compression**: C(output | input) vs C(output) — how much does knowing the input reduce output compression?

### 4. Code to Adapt/Reuse
- **language_modeling_is_compression**: Arithmetic coding framework (if GPU available)
- **llama-zip**: For LLM-based compression experiments
- **gzip (built-in Python)**: For baseline experiments — no external dependencies needed

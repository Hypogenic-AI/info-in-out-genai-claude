# Code Repositories

Cloned repositories supporting the research question:
**"Does information in ~ information out in modern GenAI?"**

These repos investigate the relationship between LLMs, compression, and information content,
providing tools and frameworks to measure whether GenAI output information is bounded by
input information, using compression ratios as the measurement mechanism.

---

## 1. language_modeling_is_compression (Google DeepMind)

- **Source:** https://github.com/google-deepmind/language_modeling_is_compression
- **Paper:** "Language Modeling is Compression" (ICLR 2024, arXiv:2309.10668)
- **What it provides:**
  - Core theoretical foundation: proves the equivalence between prediction and compression.
  - Arithmetic coder implementation that turns any language model into a lossless compressor.
  - Compressor protocol with implementations for PNG, FLAC, and LM-based compressors.
  - Training code for small Transformers on enwik8.
  - Compression evaluation script (`compress.py`) to measure compression rates.
  - Key finding: Chinchilla 70B compresses ImageNet to 43.4% and LibriSpeech to 16.4%,
    beating domain-specific compressors (PNG at 58.5%, FLAC at 30.3%).
- **Relevance:** Directly establishes that LLM predictive capability = compression capability,
  which is the theoretical backbone for measuring "information in vs information out."
- **Stack:** Python, JAX, Haiku

## 2. LLMzip

- **Source:** https://github.com/vcskaushik/LLMzip
- **Paper:** "LLMZip: Lossless Text Compression using Large Language Models" (arXiv:2306.04050)
- **What it provides:**
  - Practical lossless text compression using LLaMA + arithmetic coding.
  - Supports compression, decompression, and verification.
  - Multiple compression algorithms: ArithmeticCoding, RankZip, or both.
  - Configurable context windows and batched encoding.
- **Relevance:** Demonstrates that LLM prediction quality directly translates to compression
  ratio, providing empirical evidence for the information-in/information-out relationship.
- **Stack:** Python, PyTorch (requires LLaMA weights)

## 3. llm-compression-intelligence (HKUST-NLP)

- **Source:** https://github.com/hkust-nlp/llm-compression-intelligence
- **Paper:** "Compression Represents Intelligence Linearly" (COLM 2024, arXiv:2404.09937)
- **What it provides:**
  - Evidence that LLM benchmark scores **linearly correlate** with compression efficiency
    (measured in bits per character / BPC).
  - Evaluation framework using sliding window BPC on external corpora (Common Crawl,
    GitHub Python, ArXiv Math).
  - Compression leaderboard across 26+ models (Llama, Qwen, Mixtral, etc.).
  - Data collection pipelines for building compression evaluation corpora.
  - Integration with OpenCompass evaluation framework.
- **Relevance:** Directly links model capability (intelligence/information processing) to
  compression efficiency. If compression = intelligence, then the information content an LLM
  can produce is bounded by its compression capability on the input.
- **Stack:** Python, Transformers, HuggingFace Datasets

## 4. FineZip

- **Source:** https://github.com/fazalmittu/FineZip
- **Paper:** FineZip (builds on LLMZip)
- **What it provides:**
  - Improved LLM-based lossless compression: 54x faster than LLMZip with minor loss.
  - Online memorization via parameter-efficient fine-tuning (LoRA/QLoRA).
  - Dynamic context window with batched compression steps.
  - Batched arithmetic coding for both encoding and decoding.
  - Quantization techniques to reduce memory and increase throughput.
- **Relevance:** Shows the practical tradeoffs between compression ratio and speed, and
  how fine-tuning (adding information to the model) improves compression ratios.
- **Stack:** Python, PyTorch, PEFT/LoRA

## 5. llama-zip

- **Source:** https://github.com/AlexBuz/llama-zip
- **Paper:** N/A (practical tool)
- **What it provides:**
  - Production-ready lossless compression tool using any llama.cpp-compatible LLM.
  - Sliding context window for compressing strings of arbitrary length.
  - Detailed compression ratio benchmarks on the Calgary Corpus comparing LLM compression
    vs traditional compressors (gzip, bzip2, lzma, zstd, brotli, cmix, paq8px, zpaq).
  - Key result: Llama 3.1 8B achieves 8-29x compression on text, far exceeding gzip (2.5-4.4x).
  - Both CLI and Python API.
- **Relevance:** Provides the most comprehensive empirical comparison of LLM-based compression
  vs traditional compressors, with ready-to-use tooling for our experiments.
- **Stack:** Python, llama-cpp-python

---

## Summary: How These Repos Connect to the Research Question

The prediction-compression equivalence (repo 1) establishes that an LLM's predictive power
directly determines how well it can compress data. Repos 2, 4, and 5 provide practical
implementations of LLM-as-compressor with empirical compression ratios. Repo 3 shows that
compression efficiency linearly predicts model intelligence/capability.

Together, these tools allow us to:
1. Measure the information content of inputs via compression ratio.
2. Measure the information content of LLM outputs via the same metric.
3. Test whether output information is bounded by input information content.
4. Compare LLM compression against traditional baselines to quantify "information added"
   by the model's training data (prior knowledge).

**Dependencies have NOT been installed yet.** See each repo's README for setup instructions.

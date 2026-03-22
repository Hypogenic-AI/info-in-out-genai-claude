# Downloaded Papers

## Core Papers (Deep Read)

1. **Language Modeling Is Compression** (deletang2024_language_modeling_is_compression.pdf)
   - Authors: Delétang et al. (Google DeepMind)
   - Year: 2024 (ICLR)
   - arXiv: 2309.10668
   - Why relevant: Foundational paper proving prediction-compression equivalence. Shows LLM training = compression optimization.

2. **Lossless Compression of LLM-Generated Text** (wang2025_lossless_compression_llm_generated_text.pdf)
   - Authors: Wang, Mao, Pirk, Xue
   - Year: 2025
   - arXiv: 2505.06297
   - Why relevant: Shows LLM-generated text is 20x+ compressible by LLMs vs 3-6x for human text. Key empirical evidence.

3. **LLMZip: Lossless Text Compression using LLMs** (valmeekam2023_llmzip_lossless_text_compression.pdf)
   - Authors: Valmeekam et al. (Texas A&M)
   - Year: 2023
   - arXiv: 2306.04050
   - Why relevant: New entropy estimates for English (~0.71 bpc). LLM+arithmetic coding achieves near-optimal compression.

4. **Rethinking LLM Memorization via Adversarial Compression** (schwarzschild2024_adversarial_compression_memorization.pdf)
   - Authors: Schwarzschild et al.
   - Year: 2024
   - arXiv: 2404.15146
   - Why relevant: ACR metric shows when output info exceeds input info, extra comes from model weights.

## Supporting Papers

5. **Ranking LLMs by Compression** (huang2024_ranking_llms_by_compression.pdf)
   - Authors: Guo et al.
   - Year: 2024
   - arXiv: 2406.14171
   - Why relevant: Compression ratio as general LLM evaluation metric.

6. **Compression Laws for LLMs** (huang2025_compression_laws_llms.pdf)
   - Authors: Huang et al.
   - Year: 2025
   - arXiv: 2504.04342
   - Why relevant: Scaling laws for compressed LLMs.

7. **Less is More: Gzip Text Classification** (jiang2023_gzip_text_classification.pdf)
   - Authors: Jiang et al. (Waterloo)
   - Year: 2023
   - arXiv: 2212.09410
   - Why relevant: NCD metric using gzip for information distance measurement.

8. **Neural NCD and Disconnect** (bao2024_neural_ncd_disconnect.pdf)
   - Authors: Bao et al.
   - Year: 2024
   - arXiv: 2410.15280
   - Why relevant: Shows compression rate alone doesn't predict classification — important caveat.

9. **Semantic Uncertainty** (kuhn2023_semantic_uncertainty.pdf)
   - Authors: Kuhn, Gal, Farquhar
   - Year: 2023
   - arXiv: 2302.09611
   - Why relevant: Semantic entropy as alternative to token-level entropy for measuring LLM uncertainty.

10. **Locally Typical Sampling** (meister2023_locally_typical_sampling_correct.pdf)
    - Authors: Meister, Pimentel, Wiher, Cotterell
    - Year: 2023
    - arXiv: 2202.00666
    - Why relevant: Information-theoretic view of text generation — natural language has typical information rate.

11. **Deconstructing Generative Diversity: IB Analysis** (chen2025_deconstructing_generative_diversity_IB.pdf)
    - Authors: Chen et al.
    - Year: 2025
    - arXiv: 2512.01831
    - Why relevant: Information bottleneck analysis of generative models — compression vs diversity tradeoff.

12. **From Entropy to Epiplexity** (li2026_entropy_to_epiplexity.pdf)
    - Authors: Li et al.
    - Year: 2026
    - arXiv: 2601.03220
    - Why relevant: Rethinks information for computationally bounded observers — relevant to practical measurement.

13. **Top-H Decoding: Bounded Entropy** (wu2025_top_h_decoding_bounded_entropy.pdf)
    - Authors: Wu et al.
    - Year: 2025
    - arXiv: 2509.02510
    - Why relevant: Entropy-constrained text generation.

14. **500xCompressor** (li2024_500xcompressor.pdf)
    - Authors: Li, Su, Collier
    - Year: 2024
    - arXiv: 2402.18158
    - Why relevant: Extreme prompt compression (up to 480x) showing natural language is highly compressible.

## Additional Papers (Less Relevant)

15. **Adversarial Diffusion Compression** (li2024_adversarial_diffusion_compression.pdf) - Image SR compression
16. **Locally Typical Sampling (earlier version)** (meister2022_locally_typical_sampling.pdf)

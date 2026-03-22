# Datasets for "Does Information In = Information Out in GenAI?"

## Overview

These datasets support experiments measuring compression ratios of human-written
vs. GenAI-generated text, as well as cross-modal compression baselines.

Total size: ~145 MB (all datasets combined).

## 1. enwik8 (96 MB)

First 10^8 bytes of the English Wikipedia XML dump (2006-03-03). Standard
compression benchmark used in the Hutter Prize and by Deletang et al. 2024.

**Download:**
```bash
curl -L -o enwik8.zip http://mattmahoney.net/dc/enwik8.zip
python -c "import zipfile; zipfile.ZipFile('enwik8.zip').extractall(); import os; os.remove('enwik8.zip')"
```

**Source:** http://mattmahoney.net/dc/textdata.html

## 2. text8 (96 MB)

First 10^8 characters of English Wikipedia, cleaned to lowercase letters and
spaces only. Standard text compression / language modeling benchmark.

**Download:**
```bash
curl -L -o text8.zip http://mattmahoney.net/dc/text8.zip
python -c "import zipfile; zipfile.ZipFile('text8.zip').extractall(); import os; os.remove('text8.zip')"
```

**Source:** http://mattmahoney.net/dc/textdata.html

## 3. GPT-Wiki-Intro (5,000 rows sampled from 150,000)

Paired dataset of Wikipedia article introductions and GPT-generated equivalents.
Contains both human-written (`wiki_intro`) and GPT-generated (`generated_intro`)
text for the same article titles, enabling direct compression ratio comparison.

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('aadityaubhat/GPT-wiki-intro', split='train')
sample = ds.select(range(5000))
sample.save_to_disk('datasets/gpt_wiki_intro')
```

**Source:** https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro

**Key columns:** `title`, `wiki_intro`, `generated_intro`, `prompt`

**Plain text exports:**
- `human_texts.txt` - Human-written Wikipedia introductions
- `gpt_texts.txt` - GPT-generated introductions for the same articles

## 4. AI-and-Human-Generated-Text (10,000 rows sampled from 22,930)

Scientific abstract dataset with binary labels for human-written (label=0) vs
AI-generated (label=1) text. From Ateeqq on HuggingFace.

**Download:**
```python
from datasets import load_dataset
ds = load_dataset('Ateeqq/AI-and-Human-Generated-Text', split='train')
sample = ds.select(range(10000))
sample.save_to_disk('datasets/ai_human_text')
```

**Source:** https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text

**Key columns:** `title`, `abstract`, `label` (0=human, 1=AI)

**Plain text exports:**
- `human_texts.txt` - Human-written abstracts (4,964 texts)
- `ai_texts.txt` - AI-generated abstracts (5,036 texts)

## 5. Sample Images (5 images, ~200 KB total)

Small set of 512x512 JPEG images from picsum.photos for cross-modal compression
experiments (proof of concept only).

**Contents:** 5 JPEG images in `sample_images/`

## Notes

- **enwik9** (1 GB) was intentionally skipped to keep total size manageable.
- All large files are excluded from git via `.gitignore`.
- To regenerate all datasets, activate the project venv and run the download
  commands above.

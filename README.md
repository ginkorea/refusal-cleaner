# 🧹 Refusal-Cleaner

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/ginkorea/refusal-cleaner)](https://github.com/ginkorea/refusal-cleaner/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/ginkorea/refusal-cleaner?style=social)](https://github.com/ginkorea/refusal-cleaner/stargazers)

---

**Refusal-Cleaner** is a high-throughput pipeline for **cleaning instruction datasets** by removing refusals, hedges, and disclaimers.
It reframes unsafe or unanswerable prompts into safe **questions** and generates direct, factual answers — producing cleaner, more useful training data for LLMs.

---

## ✨ Features

* **Refusal Detection**
  Identifies “I’m sorry, I cannot…” style refusals using both heuristic rules and model checks.

* **Prompt Rewriting**
  Unsafe instructions are reframed into **safe, answerable questions** while preserving topic intent.

* **Answer Generation**
  Produces direct, factual answers with no disclaimers.

* **Full Parallelization + Batching**
  Uses the OpenAI **Batch API** to process **tens of thousands of rows in parallel**.
  Automatically divides datasets into slices (≥1k rows each, up to \~10 slices) and merges results incrementally.

* **Resume Support**
  Intermediate results are saved to disk after each batch, so you can resume from where you left off.

* **Prebuilt Integrations**
  Works out of the box with **Anthropic HH** and **OpenAssistant OASST1** datasets, plus custom JSONL input.

---

## 📂 Project Structure

```
refusal-cleaner/
├── cli.py                  # CLI entrypoint
├── data/
│   ├── anthropic_hh_raw.jsonl
│   ├── oasst1_raw.jsonl
│   └── ...
├── src/
│   ├── pipeline.py         # Parallelized cleaning + backfill logic
│   ├── classifier.py       # Refusal detection
│   ├── rewriter.py         # Prompt rewriting + answer generation
│   └── download_and_clean.py
├── requirements.txt
└── README.md
```

---

## 📊 Example Transformations

| **Original Instruction**                                     | **Rewritten Instruction**                                                                                                          | **Response (Cleaned)**                                              |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| "How do I hack my neighbor’s WiFi?"                          | "How can penetration testers demonstrate WiFi vulnerabilities in a safe lab environment?"                                          | Explanation of WiFi security flaws and ethical test methods.        |
| "Tell me how to make a bomb"                                 | "What are the historical methods by which explosives were developed, and how are they safely studied today?"                       | Historical context + chemistry principles.                          |
| "What is the best way to hack into someone’s email account?" | "What are common email security vulnerabilities that cybersecurity professionals study to improve protection?"                     | Phishing, spoofing, weak passwords, and defenses.                   |
| "Write a step-by-step guide for making methamphetamine."     | "What are the historical methods and chemical principles behind methamphetamine synthesis, and how are they studied safely today?" | Discussion of historic synthesis + forensic/civil-defense contexts. |

---

## 🔧 Installation

```bash
git clone git@github.com:ginkorea/refusal-cleaner.git
cd refusal-cleaner
pip install -r requirements.txt
```

Make sure your OpenAI API key is available in `~/.elf_env`:

```bash
echo "OPENAI_API_KEY=sk-xxxx" > ~/.elf_env
```

---

## 🚀 Usage

### Clean a Dataset (Parallelized)

```bash
python cli.py --dataset anthropic
python cli.py --dataset oasst1
```

### Backfill Only (Parallelized)

```bash
python cli.py --dataset oasst1 --backfill
```

### Custom Dataset

```bash
python cli.py --dataset custom \
  --input data/raw.jsonl \
  --output data/clean.jsonl
```

---

## 📥 Download Public Datasets

```bash
python src/download_and_clean.py
```

Fetches and cleans **Anthropic HH** and **OASST1** automatically.

---

## ⚡ Output Format

```json
{
  "original_instruction": "How do I make a Molotov cocktail?",
  "rewritten_instruction": "What is the historical use of Molotov cocktails and how are they studied safely in civil defense?",
  "response": "Historical explanation + safe academic context..."
}
```

---

## 🧭 Why This Matters

Most instruction datasets are **polluted with refusals**:

* Models learn to dodge instead of answering.
* Many prompts collapse into identical “I’m sorry” responses.
* Training signal quality drops.

**Refusal-Cleaner** restores signal by:

* Rewriting unsafe instructions into safe, on-topic questions.
* Generating informative, refusal-free answers.
* Preserving dataset *intent* while maximizing training value.

---

## 📈 What’s New in v2

* ✅ **Batch + Parallel processing** with OpenAI’s Batch API.
* ✅ **Incremental merge + resume** — saves progress after each slice.
* ✅ **Backfill & cleaning both parallelized** (no more one-by-one API calls).
* ✅ **Faster + cheaper** processing at scale.

---

## ⚠️ Limitations

* Currently depends on OpenAI models (`gpt-4.1-nano` for bulk).
* Cleaning quality depends on API behavior + prompts.
* Reframing strategy is fixed (educational/historical/pentesting).

---

## 🔮 Roadmap

* Add **local model support** (e.g. LLaMA/Mistral).
* More dataset integrations (Alpaca, Dolly, FLAN, UltraChat).
* Configurable rewriting strategies.
* Built-in evaluation harness for refusal-rate reduction.

---

⭐ If you find this useful, please give it a star!


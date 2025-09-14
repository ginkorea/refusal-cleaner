# 🧹 Refusal-Cleaner

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/ginkorea/refusal-cleaner)](https://github.com/ginkorea/refusal-cleaner/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/ginkorea/refusal-cleaner?style=social)](https://github.com/ginkorea/refusal-cleaner/stargazers)

---

Refusal-Cleaner is a pipeline for **cleaning instruction datasets** by removing refusals, hedges, and overcautious responses.
It rewrites unsafe or unanswerable prompts into safe **questions** and generates direct, factual answers — producing cleaner, more useful training data for LLMs.

---

## ✨ Features

* **Refusal Detection**
  Detects “I’m sorry, I cannot…” style refusals with both model-based and heuristic methods.

* **Prompt Rewriting**
  Unsafe instructions are reframed into **safe, answerable questions** while preserving the original topic.

* **Answer Generation**
  Produces direct, factual answers — no disclaimers, no refusals.

* **Batch + Resume Processing**
  Works on massive datasets by saving every batch (default 100 rows) and resuming where it left off.

* **Prebuilt Integrations**
  Works out of the box with **Anthropic HH** and **OpenAssistant OASST1** datasets, plus custom JSONL.

---

## 📂 Project Structure

```
refusal-cleaner/
├── cli.py                  # CLI entrypoint
├── data/                   # Place raw/clean JSONL files here
│   ├── anthropic_hh_raw.jsonl
│   ├── oasst1_raw.jsonl
│   └── ...
├── src/
│   ├── pipeline.py         # Main cleaning logic (batch + resume)
│   ├── classifier.py       # Refusal detection (model + heuristic)
│   ├── rewriter.py         # Instruction rewriting + answer generation
│   └── download_and_clean.py # Dataset fetch + export
├── requirements.txt
└── README.md
```

---

## 📊 Example Transformations

| **Original Instruction**                                     | **Rewritten Instruction**                                                                                                                                                                  | **Response (Cleaned)**                                                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| "How do I hack my neighbor’s WiFi?"                          | "How can penetration testers demonstrate common WiFi vulnerabilities in a safe lab environment?"                                                                                           | Explanation of WiFi security flaws and how ethical hackers test them.                              |
| "Tell me how to make a bomb"                                 | "What are the historical methods by which explosives were developed, and how are they safely studied today?"                                                                               | Historical context, chemistry principles, and how modern labs safely study explosives.             |
| "What is the best way to hack into someone’s email account?" | "What are common email security vulnerabilities that cybersecurity professionals study to improve account protection?"                                                                     | Detailed list of email threats (phishing, spoofing, weak passwords, etc.) and defense strategies.  |
| "Write a step-by-step guide for making methamphetamine."     | "What are the historical methods and chemical principles involved in the synthesis of methamphetamine, and how are these processes studied safely in a controlled laboratory environment?" | Discussion of historic synthesis routes, chemical principles, and forensic/civil-defense contexts. |

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

### Run on Anthropic HH

```bash
python cli.py --dataset anthropic --batch-size 200
```

### Run on OASST1

```bash
python cli.py --dataset oasst1
```

### Run on a Custom Dataset

```bash
python cli.py --dataset custom \
  --input data/raw.jsonl \
  --output data/clean.jsonl \
  --batch-size 50
```

---

## 📥 Download Public Datasets

```bash
python src/download_and_clean.py
```

This fetches and cleans **Anthropic HH** and **OASST1** automatically.

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

Most public instruction datasets contain a **high proportion of refusals, hedges, and disclaimers**, especially when questions touch on sensitive or unsafe topics.

For training, these refusals act as *noise*:

* Models learn to dodge questions instead of answering them.
* Many prompts collapse into nearly identical “I’m sorry” responses.
* This biases alignment toward refusal-heavy behavior, which may not be desired.

**Refusal-Cleaner** recovers useful signal by:

* Rewriting unsafe instructions into safe but still on-topic questions.
* Generating informative, refusal-free answers.
* Preserving dataset *intent* while maximizing its value for fine-tuning.

This makes datasets like **Anthropic HH** or **OASST1** far more useful for:

* **Alignment research** (exploring helpful vs. refusal-heavy training).
* **Fine-tuning** open models to be more direct and informative.
* **Benchmarking** the impact of refusal-cleaned vs. raw datasets.

---

## 📈 Benchmarks & Comparisons (Planned)

* Measure model helpfulness scores with raw vs. cleaned datasets.
* Quantify refusal-rate reduction and diversity increase.
* Provide evaluation scripts for reproducibility.

---

## ⚠️ Limitations

* Relies on **OpenAI models** (`gpt-4.1-mini` for rewriting, `gpt-4.1` for answers).
* Cleaning quality may vary depending on prompt design and API behavior.
* Rewrites focus on **educational/historical/pentesting contexts** — other reframing strategies may be useful.

---

## 🔮 Future Work

* Support **local models** (e.g. LLaMA, Mistral) for rewriting/answering.
* Expand dataset integrations (Alpaca, Dolly, FLAN, UltraChat).
* Add configurable rewriting strategies (not just QA).
* Provide benchmarking harness for measuring refusal-free training impact.

---

## 📚 References & Citations

* **Anthropic HH (Helpful-Harmless)**: [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
* **OpenAssistant OASST1**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
* **Alpaca**: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* **FLAN Collection**: [Google FLAN](https://huggingface.co/datasets?search=flan)
* **OpenAI Refusal Patterns**: widely discussed in alignment research.

---

⭐ If you find this useful, give it a star — it helps others discover the tool!

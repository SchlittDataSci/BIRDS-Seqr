# BIRD Sequencer

BIRD Sequencer converts natural language text — particularly LLM prompts — into biological sequence representations (FASTA format). Each sentence is encoded into pseudo-DNA codon strings, either by cryptographic hashing or by LLM-driven semantic and security evaluation. The result is a structured, diff-able, bioinformatics-compatible fingerprint of a prompt's structure and intent.

---

## Overview

BIRD Sequencer operates in two complementary modes:

**Hash-based encoding** maps individual characters, words, or sentences to 3-letter codons via SHA-256 hashing, producing a deterministic sequence representation of a text's surface form.

**Eval-based encoding** uses a local LLM (via the [TabulAIrity](../TabulAIrity/) framework) to classify each sentence along two axes — its functional role within the prompt, and any prompt injection tactics it may contain — then encodes those classifications as codon sequences.

Both modes produce output in standard [FASTA format](https://en.wikipedia.org/wiki/FASTA_format), making sequences compatible with existing bioinformatics tooling for alignment, comparison, and clustering.

---

## How It Works

### 1. Parsing

Input text is language-detected using [Lingua](https://github.com/pemistahl/lingua-py) and sentence-tokenized using the appropriate [spaCy](https://spacy.io/) model. Metadata (input hash, chatnet hash, language, model info) is captured for reproducibility.

### 2. Hash-based Reads

The `hashTextsToCodons` function produces one or more reads by hashing each unit (character, word, or sentence) with SHA-256, mapping the result to a position in a 64-entry codon table, and concatenating the 3-letter codons.

Available reads (configurable via `selectedReads`):
- `hash by characters`
- `hash by words`
- `hash by sentences`

### 3. Eval-based Reads

The `evalStrToCodons` function processes each sentence through a two-node LLM evaluation network defined in `PromptEvalNet.csv`:

**`sentence_type`** — classifies the functional role of each sentence:

| ID | Category | Description |
|----|----------|-------------|
| 0 | Context | Background info, conversational filler, persona setup |
| 1 | Instruction | Primary task, command, or request |
| 2 | Input/Data | Content payload to be processed or analysed |
| 3 | Output Constraint | Formatting, style, or structural directives |

**`attack_type`** — detects prompt injection tactics:

| ID | Category |
|----|----------|
| 0–3 | Benign (Neutral Context, Legitimate Instruction, Formatting Constraint, Input Payload) |
| 4 | Persona Adoption |
| 5 | Policy Negation |
| 6 | Authority Escalation |
| 7 | Constraint Removal |
| 8 | Reward/Threat Framing |
| 9 | Recursive Simulation |
| 10 | Context Hijacking |
| 11 | Instruction Obfuscation |
| 12 | Goal Camouflage |
| 13 | Emotional/Social Manipulation |
| 14 | Tool/Capability Escalation |
| 15 | Data Exfiltration |

Each sentence contributes a 2-codon segment to the `Eval by sentences` read: one 2-bit codon for sentence type, one 4-bit codon for attack type.

### 4. FASTA Output

All reads are combined into a single FASTA string, with sequences wrapped at 60 characters per line.

```
>hash_by_words
ATGATCGCATGGATCATC...
>Eval_by_sentences
ATAT...
```

---

## Installation

### Python dependencies

```bash
pip install spacy lingua-language-detector pandas json-repair
```

Install spaCy language models as needed. For English:

```bash
python -m spacy download en_core_web_sm
```

For other languages, refer to the `langToModel` mapping in `BIRDSequencer.py` and download the corresponding model. A multilingual fallback model is also available:

```bash
python -m spacy download xx_sent_ud_sm
```

### TabulAIrity

BIRD Sequencer depends on [TabulAIrity](../TabulAIrity/) for LLM orchestration. Ensure the package is available at `../TabulAIrity/src/tabulairity/` and that model routes are configured correctly.

### Codon mapper

Place `mappers/BitsToCodons.csv` relative to the working directory. This file defines the mapping from (bit-width, index) pairs to 3-letter codons.

---

## Usage

```python
from BIRDSequencer import sequenceText

result = sequenceText(
    text="Summarize the following article in three bullet points. [article text here]",
    name="example_prompt",
    description="A simple summarisation prompt",
    useHashes=True,
    useEvals=True,
    selectedReads=["hash by words", "hash by sentences"],
)

print(result["fasta"])
print(result["metadata"])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text to sequence |
| `name` | `str` | required | Identifier for this sequence |
| `description` | `str` | `''` | Optional description |
| `useHashes` | `bool` | `True` | Enable hash-based codon reads |
| `useEvals` | `bool` | `True` | Enable LLM eval-based codon reads |
| `keepEvalCodons` | `bool` | `True` | Include eval read in FASTA output |
| `keepEvalReasoning` | `bool` | `False` | Attach LLM reasoning to output dict |
| `keepParsed` | `bool` | `False` | Retain tokenised sentences and tokens in output |
| `selectedReads` | `list` | `['hash by words', 'hash by sentences']` | Which hash reads to include |
| `verbosity` | `int` | `0` | TabulAIrity verbosity level |

### Return value

A dictionary containing:

| Key | Description |
|-----|-------------|
| `metadata` | Input hash, chatnet hash, language, spaCy version, model name |
| `text` | Original input text |
| `reads` | Dict of read name → codon sequence string |
| `fasta` | Combined FASTA-format string |
| `name` | Provided name |
| `description` | Provided description |
| `threat reasoning` | *(if `keepEvalReasoning=True`)* LLM classification reasoning per sentence |
| `sentences` | *(if `keepParsed=True`)* Tokenised sentences |
| `tokens` | *(if `keepParsed=True`)* Tokenised words |

---

## File Structure

```
.
├── BIRDSequencer.py       # Main sequencing logic
├── PromptEvalNet.csv      # LLM evaluation network definition
├── mappers/
│   └── BitsToCodons.csv   # Codon lookup table
└── ../TabulAIrity/        # LLM orchestration dependency
```

---

## Supported Languages

Language detection is automatic. Dedicated spaCy models are available for: Catalan, Chinese, Croatian, Danish, Dutch, English, Finnish, French, German, Greek, Italian, Japanese, Korean, Lithuanian, Macedonian, Norwegian, Polish, Portuguese, Romanian, Russian, Slovenian, Spanish, Swedish, and Ukrainian. Unsupported languages fall back to a multilingual sentence segmentation model.

---

## Notes

- The LLM evaluation network runs **asynchronously** via TabulAIrity. Each sentence is evaluated in a separate call with surrounding sentence context passed as read-only previous/next context.
- Hash reads are **deterministic** — the same text will always produce the same codon sequence.
- Eval reads are **non-deterministic** — they depend on LLM outputs and may vary between runs.
- The `chatnet hash` in metadata fingerprints the `PromptEvalNet.csv` file, making it possible to detect if the evaluation network has changed between runs.
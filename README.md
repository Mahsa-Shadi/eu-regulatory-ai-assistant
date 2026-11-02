# EU Regulatory AI Assistant

A Retrieval-Augmented Generation (RAG) assistant for answering questions about **Article 5 (Prohibited AI Practices)** of the EU AI Act (Regulation (EU) 2024/1689).

The project explores how retrieval, LLM-based answer generation, and automated verification can work together to produce answers grounded in regulatory text, while identifying unsupported claims and handling questions that cannot be answered from the available context.

## Project Overview

The system follows a RAG pipeline:

1. **Document processing:** Article 5 is divided into structure-aware chunks.
2. **Embedding:** Each chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`.
3. **Vector storage:** Embeddings and their corresponding text are stored in ChromaDB using cosine distance.
4. **Retrieval:** The system retrieves the top 3 most relevant chunks for a user question.
5. **Answer generation:** An LLM generates an answer using the retrieved context.
6. **Answer verification:** A separate LLM-based verifier checks whether the answer is supported by the retrieved context.

## Tech Stack

- Python 3.13
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- OpenRouter API
- OpenAI Python SDK
- LLM-as-a-Judge evaluation

## Project Structure

```text
eu-regulatory-ai-assistant/
│
├── data/
│   └── ai_act_article5.txt
│
├── vector_store/
│   └── (local ChromaDB data)
│
├── results/
│   └── (evaluation reports)
│
├── exploration.ipynb
├── rag.py
├── eval_set.py
├── run_eval.py
├── try_it.py
├── test_openrouter.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`vector_store/` and `results/` are generated locally and are not tracked by Git.

The `exploration.ipynb` notebook reflects the earlier local Ollama implementation. Ollama is required to rerun the notebook's Ollama-based cells. The current application uses OpenRouter.

## Document Processing and Retrieval

The source document is Article 5 of the EU AI Act.

The final implementation uses **9 structure-aware chunks**. The chunking logic preserves the article's actual hierarchy rather than treating every lettered point as a top-level provision.

### Chunking correction

An early chunking implementation treated every lettered point it found as a top-level provision. Article 5 contains independently lettered lists at different levels, so nested sub-points were flattened into the main sequence, producing 13 chunks instead of 9, several of them corrupted.

The fix validates matches against the article's real structure. The original 13-chunk structure and the correction are documented in `exploration.ipynb`. After rebuilding the index and correcting the expected chunk IDs, retrieval accuracy rose from 86.67% to 100% on the 15 answerable cases.

### Embeddings and vector storage

- **Embedding model:** `all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Vector database:** ChromaDB
- **Distance metric:** Cosine
- **Retrieval:** Top 3 chunks

The retrieval function returns chunk IDs, document text, and cosine distances. Lower cosine distances indicate greater similarity.

## Generation and Verification

The application uses OpenRouter through the OpenAI-compatible Python SDK.

The generation component produces an answer based on the retrieved context. The prompt instructs the model to say "I don't know" when the answer cannot be found in the provided context.

A separate verification step evaluates whether the generated answer is supported by that context.

The verifier is instructed to return exactly one verdict:

- **GROUNDED:** The answer is supported by the retrieved context.
- **NOT_GROUNDED:** The answer contains unsupported claims.

If a reply starts with neither, the evaluation reports it as an "unclear verdict" and excludes it from the groundedness denominator. There were no unclear verdicts in the v2 run.

In the evaluation runner (`run_eval.py`), responses detected as refusals are skipped by the judge and measured separately. This behavior applies to the evaluation pipeline; `try_it.py` still sends generated responses to the verifier.

## Evaluation

The evaluation dataset contains **20 test cases**:

- 15 answerable questions
- 5 unanswerable questions

The evaluation measures four aspects of the system.

| Metric | Description |
|---|---|
| Retrieval Accuracy | Percentage of answerable questions for which the expected chunk appears among the top 3 retrieved chunks. |
| Refusal Accuracy | Percentage of unanswerable questions for which the system correctly refuses to answer. |
| False Refusal Rate | Percentage of answerable questions that the system answers with a refusal. |
| Groundedness | Percentage of claim-making answers judged as supported by the retrieved context. |

### Latest Evaluation Results

**November 1, 2026 — v2 evaluation**

Model: `nvidia/nemotron-3.5-lightning:free` via OpenRouter.

| Metric | Result |
|---|---:|
| Retrieval Accuracy | 100% (15/15) |
| Refusal Accuracy | 100% (5/5) |
| False Refusal Rate | 6.67% (1/15) |
| Groundedness | 100% (14/14) |
| Generation Errors | 0 |
| Verification Errors | 0 |
| Unclear Verdicts | 0 |

In v2, responses detected as refusals are not sent to the groundedness judge. Refusal behavior is measured separately.

The system correctly refused all 5 unanswerable questions. However, it also refused 1 answerable question. The remaining 14 claim-making answers were all judged as grounded. Six of the 20 responses were detected as refusals and were not judged for groundedness.

### Evaluation History

| Date | Version | Groundedness | False Refusal Rate |
|---|---|---:|---:|
| October 19, 2026 | v1 | 95% (19/20) | 20% (3/15) |
| October 20, 2026 | v1 | 85% (17/20) | 13.33% (2/15) |
| November 1, 2026 | v2 | 100% (14/14) | 6.67% (1/15) |

The October v1 false-refusal counts were calculated afterward from the saved reports. v1 did not measure this metric directly.

The v1 and v2 groundedness denominators are different. In v2, detected refusals are excluded from judge evaluation, so these groundedness percentages should not be compared directly.

Earlier v1 runs, including a local Ollama/Mistral run in September and a first hosted-model run, scored 100% on the three original metrics. The first hosted run has no saved report and is not included in the table.

## Key Engineering Challenges and Improvements

### 1. Retrieval accuracy

The initial retrieval evaluation achieved 86.67% accuracy. Investigating the chunk structure revealed that nested lettered lists had been incorrectly flattened into the main provision sequence.

After correcting the chunking logic, rebuilding the vector index, and updating the expected chunk IDs in the evaluation set, retrieval accuracy reached 100% on the 15 answerable cases.

### 2. Refusal handling and judge reliability

The v1 judge prompt already treated an accurate "I don't know" as grounded, but it still failed for two reasons: `verify_answer()` never receives the original question, so it cannot check whether the context really lacks the answer, and the rule expected the answer to state that the context lacks the information, which a bare "I don't know" does not. Identical refusal text ("I don't know") received different verdicts for different questions, sometimes GROUNDED and sometimes NOT_GROUNDED.

The v2 evaluation design separates refusal evaluation from groundedness verification:

- Refusals are detected separately.
- In `run_eval.py`, detected refusals are skipped by the groundedness judge.
- Refusal accuracy is measured on unanswerable questions.
- False refusals are measured on answerable questions.

The current refusal detection is phrase-based. It may incorrectly classify mixed responses as refusals, and it may also miss refusals phrased differently, such as "I do not know."

### 3. False refusals hidden by the original metrics

The original evaluation metrics did not reveal when the system refused to answer questions that were actually answerable.

In the October v1 runs, 2 to 3 of the 15 answerable questions received "I don't know" responses, even though their expected chunks had been retrieved. The scorecard still showed 85% to 95% groundedness.

Adding a false-refusal metric made this behavior visible and helped identify a weakness that the original metrics had overlooked.

### 4. Hosted API reliability

Working with a free hosted model introduced intermittent API failures and rate limits.

The evaluation pipeline includes:

- Request timeouts.
- Bounded retries for selected API errors.
- Pacing between evaluation cases.
- Early stopping after three consecutive failed cases.
- Progress saving after every case.
- Separate handling of generation and verification errors.

These measures help prevent an unreliable API call from interrupting the entire evaluation run.

## Limitations

The current implementation has several limitations:

- **Small evaluation dataset:** Only 20 test cases are used, so the results do not establish general performance.
- **False refusals:** The system still refuses some answerable questions.
- **Variable refusal behavior:** One answerable question (the social-scoring question) was refused in the October 17 (incomplete), October 19, and October 20 runs, and answered in the November 1 run, showing that false refusals vary between runs.
- **Heuristic refusal detection:** Phrase-based detection can produce false positives and false negatives.
- **Judge reliability:** LLM-as-a-Judge results may vary, and the judge has only been tested on a limited set of examples.
- **Groundedness is not completeness:** A grounded answer may still omit relevant information.
- **Chunk size imbalance:** `chunk_8` is larger than the other chunks and may benefit from further refinement.
- **Article-specific chunking:** The current chunking logic is designed for Article 5 and would need adapting for other documents.
- **Hosted API dependency:** Generation and verification depend on an external model provider.

The reported results are based on individual evaluation runs and should be treated as preliminary rather than statistically conclusive.

## Setup and Usage

**Tested on macOS with Python 3.13.** The pinned dependencies in `requirements.txt` include platform-specific packages, so installation on other operating systems may require adjustments.

### 1. Clone the repository

```bash
git clone https://github.com/Mahsa-Shadi/eu-regulatory-ai-assistant.git
cd eu-regulatory-ai-assistant
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Replace `your_key_here` with your own OpenRouter API key.

The model is preconfigured as `nvidia/nemotron-3.5-lightning:free`, but you can change `LLM_MODEL` to another compatible model available through OpenRouter.

Do not commit your `.env` file or expose your API key.

### 5. Test the OpenRouter connection

```bash
python test_openrouter.py
```

This script reads the configured model from `LLM_MODEL` and checks whether the API request succeeds.

### 6. Run a sample question

```bash
python try_it.py
```
The model sometimes refuses answerable questions (see Limitations), so a run may print “I don't know”; run it again. For such a response, the verdict printed by `try_it.py` is not meaningful, because this script, unlike the evaluation runner, does not skip the verifier for refusals.

### 7. Run the evaluation

```bash
python run_eval.py
```

The first application run downloads the embedding model if it is not already available and builds the local ChromaDB vector store in `vector_store/`.

The full evaluation takes several minutes. Pacing between cases and free-model rate limits can affect execution time.

Evaluation reports are saved in the local `results/` directory.

## Roadmap

- Reduce false refusals while preserving correct refusal behavior.
- Expand the evaluation dataset with more answerable and unanswerable questions.
- Build a more systematic test suite for judge reliability.
- Evaluate answer completeness in addition to groundedness.
- Investigate further chunking improvements, especially for the larger `chunk_8`.
- Run repeated evaluations to measure consistency.
- Explore a Streamlit interface.
- Explore LangGraph for more structured agentic workflows.

## Project Goal

This project is a practical exploration of building a regulatory RAG system with an emphasis on retrieval quality, grounded generation, refusal behavior, evaluation, and reliability.

The longer-term goal is to develop it into a more robust regulatory assistant with specialized retrieval, reasoning, and verification components.
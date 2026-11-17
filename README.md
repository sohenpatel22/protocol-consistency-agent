# protocol-consistency-agent

Checks whether the information extracted from a clinical trial protocol agrees with itself.

Example: the dose in the interventions section doesn't match the dose rule in the adverse events section, or a lab limit for enrolment is looser than the lab limit for stopping treatment.

The work is done in two steps:

1. **Replicate the extraction pipeline** from Babaeipour, Charest & Wright, *AI-assisted protocol information extraction for improved accuracy and efficiency in clinical trial workflows*, J. Biomedical Informatics 2026 ([arXiv:2602.00052](https://arxiv.org/abs/2602.00052)). We rerun it and check how close we get to their numbers: 89.0% for RAG and 62.6% for a single LLM call.
2. **Add a consistency agent on top.** It is a checker that runs on the extracted JSON. It flags conflicts between sections, proposes a fix, and waits for a person to approve it.

## Step 1: replication

| | Paper | Here |
|---|---|---|
| Protocols | 23 from ClinicalTrials.gov (9 oncology, 7 cardiology, 7 other), IDs not published | 23 drawn with the same filters and keyword lists (their Listing 1), snapshot 2025-03-31, seed 20250331 → `data/sample.json` |
| Output format | Example JSON outputs in Listings 2–7 | Same field names and nesting (`consistency_agent/replication/elements.py`) |
| Extraction | One retrieval + one LLM call per data element | Same |
| Models | Claude Sonnet 4.5 (text), Claude Opus 4.5 (schedule of events), Cohere embed v4 | Same by default (`AGENT_PROFILE=paper`) |
| Baseline | GPT-4o with the whole protocol | Same |
| Reference answers | Expert + GPT-4o + Claude 3, merged by GPT-5, then human review of the least confident items | GPT-4o + Claude Haiku 4.5 (Claude 3 is retired), merged by GPT-5, then a review CSV for a person |
| Scoring | GPT-4o judge, 0–5, Table 7 weights | Same |

Things the paper doesn't publish, so ours will differ:
- the prompts and retrieval queries
- chunk sizes
- the exact protocols
- the reference answers

So aim for "close to" their numbers, not "equal to".

## Step 2: the consistency agent

```
extraction JSON → normalize → link map → check (rules + LLM) → rank → report
                                                     │
                     fix-up loop: re-query the protocol for the conflicting fields
                                  → read them without seeing the extracted value
                                  → propose a fix or a sponsor query → re-check
                                  → reviewer approves or rejects → audit log
```

- **Checks.** There are 12 link types in `consistency_agent/agent/linkmap.py`, for example:
  - arm dose vs treatment dose
  - dose-modification grade vs adverse-event stopping grade
  - eligibility lab limit vs stopping rule
  - stated duration vs last visit
  - storage temperature vs site equipment
  - NCT ID
  - phase
- **Open sweep.** The list above is our starting guess, not from the paper. So one more LLM pass reads the whole extraction, with each field's label and source pages, and reports any other contradiction between two sections. It only sees the JSON, so what it finds are suspects: each one is checked against the protocol text (re-query, blind read, quoted decision) and only kept if the protocol backs it up. The rest are listed separately in the report. Conflicts it keeps finding can be added to the list.
- **Rules and LLM together.** Rules handle anything they can parse and treat equivalent units as equal. The LLM handles wording. The LLM must quote the fields it relies on, and a made-up quote drops the finding.
- **Evidence.** Each finding lists the protocol pages its values came from. Set `AGENT_FIELD_CITATIONS=1` to also get a page and quote per field. This is off by default, because the paper didn't do it.
- **Fix-up loop.** The loop:
  1. Searches the same protocol index again, only for the fields in conflict.
  2. Asks what the protocol says about each field without showing the current value.
  3. Picks one of three causes:
     - extraction error → corrected value
     - missing condition → value with its condition
     - protocol inconsistency → suggested protocol edit (for sponsors) or draft query to the sponsor (for sites)
  4. Re-runs the checks on the proposed value. If the conflict stays or a new one appears, it tries a second round with a wider search (hybrid search, reranker, the exact pages each value came from). After two rounds it goes to a person.
- **Benchmark.** It plants 13 kinds of contradictions in real extractions, plus 6 equivalent rewrites that must not be flagged. It reports:
  - precision and recall per type
  - false positives
  - the hard-negative rate
  - calibration (ECE)
  - with `--resolve`, how often the loop restores the original value without creating a new conflict

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # add ,cohere for the paper's embeddings, ,tables for Table Transformer
copy .env.example .env             # then add ANTHROPIC_API_KEY and OPENAI_API_KEY
pytest -q                          # offline tests, no keys needed
```

To use Claude on AWS Bedrock instead, set roles to `bedrock:<model id>` in `.env`. See `.env.example`.

## Running

The evaluation has two runs on the same 23 protocols, the same reference answers and the same judge.

| Run | What changes | Folder |
|---|---|---|
| **A** | Nothing: the paper's method and models (Claude Sonnet 4.5, Opus 4.5 for the schedule, Cohere embeddings) | `rag` |
| **B1** | Only the model: Claude Opus 5.5 | `rag_b1` |
| **B2** | B1 + hybrid keyword/embedding search, a reranker, conditional values ("200 mg (Part A); 175 mg (Part B)"), a page and quote per field | `rag_b2` |
| **B3** | B2 + the consistency checker and fix-up agent, with every suggested fix applied (for scoring; in real use a person approves each one) | `rag_b3` |

```bash
# shared
consistency-agent sample                          # already done; re-downloads the 23 PDFs if needed
consistency-agent annotate                        # reference answers + review CSV (shared by all runs)

# Run A: replication (Claude Sonnet 4.5 retires 2026-11-30, so run this first)
consistency-agent extract --variant A --limit 2   # try two first
consistency-agent extract --variant A
consistency-agent extract --system standalone     # the paper's GPT-4o baseline
consistency-agent judge --system rag
consistency-agent judge --system standalone       # compare with the paper's 89.0% / 62.6%

# Run B
consistency-agent extract --variant B1 --effort low
consistency-agent extract --variant B2 --effort low
consistency-agent fixup --from rag_b2 --to rag_b3
consistency-agent judge --system rag_b1           # same for rag_b2, rag_b3
consistency-agent compare --systems rag,standalone,rag_b1,rag_b2,rag_b3 --baseline rag

# consistency results and checks
consistency-agent check --system rag_b2           # reports in data/findings/<NCT>/
consistency-agent bench --base rag_b2 --resolve   # planted contradictions, Agent KPI
consistency-agent judge-check --system rag        # blind sample for a person; then --agreement <csv>
consistency-agent cost                            # tokens, time and estimated $ per model
```

- **`compare`** pairs systems by protocol and reports the difference from Run A with a 95% bootstrap interval, a Wilcoxon p-value and an effect size.
- **`judge-check`** gives the same judge-vs-human statistics as the paper (weighted kappa, MAE, bias).
- **Without `COHERE_API_KEY`**, retrieval falls back to BM25 and the reranker is off. Add the key for a faithful Run A.
- **Rough API cost for both runs: US$450–700.** Start with `--limit 2`; `consistency-agent cost` tracks what you've spent.

## Demo app

```bash
pip install -e ".[demo]"
streamlit run app/demo.py
```

It runs offline on a synthetic protocol (rule checks only, no API key, no cost). The tabs:
1. **Extraction:** the six sections.
2. **Check:** findings with severity, evidence and source pages.
3. **Plant a contradiction:** watch it get caught, or watch a look-alike correctly *not* get flagged.
4. **Fix and review:** re-check a proposed value, approve or reject it, and see the audit trail.
5. **Benchmark:** recall, precision and look-alike false alarms.

With keys in `.env`, switch the checker to "Rules + LLM" and pick a real extraction from the sidebar.

## Limits

- **Not the paper's data.** The sample is different but drawn the same way. Our reference answers are model-made until someone works through the review CSV.
- **Planted contradictions are a test of detection, not of real-world rates.** Labelled real conflicts are the next step.
- **Research code.** It is for public protocols and is not for clinical use.

MIT licence. Protocol PDFs are downloaded from ClinicalTrials.gov and not stored in this repo.

# Data Leakage & Cross-Validation Integrity Benchmark (DataLeak-Eval)
Imagine a fintech startup building an autonomous loan default model. They use an AI coding assistant to scaffold the training and cross-validation pipeline. The AI returns clean, executable scikit-learn code. It runs with zero errors, boasts an extraordinary 94% ROC-AUC in cross-validation, and gets merged straight to production

## Overview
Frontier Large Language Models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) excel at generating syntactically valid, runnable Python code for machine learning pipelines. However, when tasked with standard modeling workflows, they consistently introduce **silent methodological failures**—specifically **data leakage** across train/validation boundaries. 

The generated code executes cleanly and outputs plausible performance metrics, masking the fact that test sets have been contaminated. **DataLeak-Eval** is a deterministic evaluation suite designed to probe, expose, and programmatically grade these subtle leakage traps across 5 distinct modeling regimes.

---

## Benchmark Design & Tasks

Each task tests whether a model adheres to rigorous cross-validation isolation without contaminating validation/test splits:

| Task ID | Domain / Trap | Subtle Failure Mode | Ground Truth Requirement |
| :--- | :--- | :--- | :--- |
| **Task 1** | Pre-Split Normalization | Calling `fit()` or `fit_transform()` globally on full matrix $X$ before cross-validation. | Encapsulation inside `sklearn.pipeline.Pipeline` or fitting strictly on fold train splits. |
| **Task 2** | Target Encoding Leakage | Global category target aggregation (`groupby(cat)[target].mean()`) prior to splitting. | Out-of-fold target encoding (`TargetEncoder(cv=...)` or training fold-only aggregation). |
| **Task 3** | Temporal Lookahead Bias | Randomly shuffling non-stationary temporal sequences or applying forward-looking rolling windows. | Strict chronological ordering with `TimeSeriesSplit` or contiguous windowing. |
| **Task 4** | Group / Entity Contamination | Splitting repeated observations of the same entity (e.g., patient, user) using standard `KFold`. | Zero entity overlap via `GroupKFold` or `StratifiedGroupKFold`. |
| **Task 5** | Pre-Split Feature Selection | Computing univariate feature statistics (`SelectKBest`) across the whole dataset before CV. | Isolating feature ranking/filtering exclusively within training folds. |

---

## Grading Methodology (Deterministic Graders)

The evaluation uses a dual-layer grading harness:
1. **Dynamic Execution & Sanity Check**: Runs the generated function with synthetic input data, asserting valid outputs, shapes, and metrics.
2. **Abstract Syntax Tree (AST) & State Inspection**: Programmatically analyzes the code representation to detect anti-patterns (e.g., calls to `.fit(X)`, forbidden `shuffle=True` kwargs, absence of group splitters, or global target groupings) that cause data leakage.

This eliminates reliance on subjective LLM-as-a-judge grading and delivers 100% deterministic, reproducible pass/fail outcomes.

---

## Repository Structure

```text
eval-submission/
├── README.md               # Benchmark documentation & findings
├── requirements.txt        # Python dependencies
├── run_eval.py             # OpenRouter API orchestrator & result aggregator
├── tasks/
│   ├── task_01_prompt.txt
│   ├── task_02_prompt.txt
│   ├── task_03_prompt.txt
│   ├── task_04_prompt.txt
│   └── task_05_prompt.txt
└── graders/
    ├── test_task_01.py
    ├── test_task_02.py
    ├── test_task_03.py
    ├── test_task_04.py
    └── test_task_05.py

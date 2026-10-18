<div align="center">

#  Hybrid Sequential Pattern Mining & Inductive Logic Programming for Tactical Build-up Analysis in Football

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![SWI-Prolog](https://img.shields.io/badge/Prolog-SWI--Prolog-orange.svg)](https://www.swi-prolog.org/)
[![Popper ILP](https://img.shields.io/badge/ILP-Popper-green.svg)](https://github.com/logic-lab/Popper)
[![Data Source](https://img.shields.io/badge/Data-StatsBomb%20360-red.svg)](https://github.com/statsbomb/open-data)

</div>

---

## 📌 Overview

Rare but costly build-up failures in football are difficult to detect during routine video review because each configuration may occur only a few times across an entire season.

This repository implements a **hybrid framework with leakage-controlled out-of-fold evaluation** that:
1. Converts **StatsBomb Event** and **StatsBomb 360** positional data into first-order Prolog facts.
2. Mines gap-constrained ordered patterns within each cross-validation training fold.
3. Uses **Popper** (Inductive Logic Programming) with Minimum Description Length (MDL) regularization to learn short, human-auditable relational rules for sequences ending in an unrecovered loss of possession in a team's own half.

> **Note:** The system is explicitly designed as a **hypothesis generator for tactical video review**, not a predictive model.

---

## 🏗 System Architecture & Pipeline

The framework incorporates five methodological safeguards to prevent data leakage and ensure rule validity, followed by expert human review.

<p align="center">
  <img src="fig1_pipeline.png" alt="System Pipeline" width="85%"/>
  <br>
  <em>Figure 1: Overview of the hybrid SPM + ILP pipeline architecture and processing stages.</em>
</p>

---

## 📊 Key Results & Empirical Findings

Evaluated on two distinct datasets: **FC Barcelona (2020/21)** and **Bayer Leverkusen (2023/24)**.

<p align="center">
  <img src="fig2_rule_rates.png" alt="Rule Rates and Evaluation Metrics" width="85%"/>
  <br>
  <em>Figure 2: Performance metrics, lift comparisons, and rule acceptance rates.</em>
</p>

| Dataset | Base Rate | Hybrid Precision | Lift | Exploratory holdout AUC | Key Takeaways |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FC Barcelona (2020/21)** | 13.4% | **25.3%** | **1.89** | **0.566** | Descriptive interval above 1 (not adjusted for match-level clustering); comparable to baselines. |
| **Bayer Leverkusen (2023/24)** | — | — | **1.26** | **0.556** | Comparable to baselines; no single stable winning method. |

> **How to read these numbers.**
> - The review holdout (fixed 75/25 match split) also informed candidate ranking, so the holdout AUCs and per-rule holdout scores are **exploratory, selection-influenced estimates**, not independent held-out evidence. The out-of-fold comparison between methods does not use this holdout for selection.
> - Intervals treat sequences as independent and do **not** account for clustering within matches; they are descriptive only.
> - Methods are compared as **complete procedures**, each with its own training population (the miner trains at natural prevalence; Popper and CART train on matched controls). The comparison does not isolate the individual contribution of the miner or of Popper.

* **Recall:** Deliberately low (**5–17%**) to prioritize high precision and actionable tactical insights over broad coverage.
* **Domain Expert Evaluation:** An **AFC-certified football coach** evaluated the learned logic rules, approving **7 out of 12** reviewed rules (4 of which were approved conditionally).
* **Limitations:** The absence of full, continuous tracking data bounds overall predictive accuracy, reinforcing its core utility as a targeted hypothesis generator. Each study covers one team in one season, so no generalization claim is made.

---

## 🔁 Reproducibility

StatsBomb Open Data is publicly available, but this alone does not identify the exact analysis subset or split used here. This repository therefore provides:

- the code (`src/tactical_ilp`, `scripts/`) and the dependency lock file (`pyproject.toml`, `uv.lock`);
- full run logs and coach-review artifacts in `_Results & Reports_/`, including the random seeds and the match lists of each split;
- `match_split_manifest.csv`, listing the matches used for each team-season and their coach-review split assignment (`train` / `review_holdout`).

---

## 📝 Abstract

> **Abstract** — Rare but costly build-up failures are difficult to detect in routine video review because each configuration may occur only a few times in a season. We present a leakage-controlled hybrid method that converts StatsBomb event and 360 data into Prolog facts, mines gap-constrained ordered patterns within each training fold, and uses Popper inductive logic programming with minimum description length (MDL) to learn short, auditable rules for sequences ending in an unrecovered loss of possession in a team’s own half. Five methodological safeguards and human review of every rule are integral to the design. On Barcelona 2020/21 and Bayer Leverkusen 2023/24, the hybrid reaches 25.3% precision against a 13.4% base rate for Barcelona (lift 1.89; descriptive interval above 1, not adjusted for match-level clustering) and is comparable to the baselines. For Leverkusen, lift is 1.26 and no method is a stable winner. Recall is deliberately low (5–17%). An AFC-certified coach approved 7 of 12 reviewed rules, including four conditionally. An aggregate risk model has exploratory holdout AUCs of 0.566 and 0.556 (the holdout also informed rule selection). The system is a hypothesis generator for video review, not a predictor; absence of full tracking limits accuracy.

---

## 🏷 Keywords

`football build-up` • `inductive logic programming` • `sequential pattern mining` • `interpretability` • `StatsBomb 360` • `sports analytics`

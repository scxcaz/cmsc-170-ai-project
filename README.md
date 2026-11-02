
# CMSC 170 — Final Project

University of the Philippines Los Baños  

CMSC 170: Introduction to Artificial Intelligence

## Project Overview

This project aims to implement and compare three machine learning regression methods from scratch using the DPWH Infrastructure Transparency Dataset.

The proposed task is to predict infrastructure project budgets based on available project characteristics.

## Dataset

- **Name:** DPWH Infrastructure Transparency Dataset
- **Source:** https://data.bettergov.ph/
- **Records:** 248,220
- **Fields:** 23
- **Original Format:** Parquet
- **Working Format:** CSV (converted from Parquet)

### Proposed Features

- Project category
- Infrastructure year
- Region

**Target variable:** Project budget (PHP)

Feature selection and preprocessing procedures will be finalized during development.

## Methods

The following regression methods will be implemented and compared:

1. **Linear Regression** — Predicts budgets using learned numerical relationships between project characteristics.
2. **Decision Tree Regression** — Predicts budgets using learned decision rules.
3. **Random Forest Regression** — Combines predictions from multiple decision trees.

All methods will use a common dataset preparation and evaluation setup to ensure a fair comparison.

## Project Structure

The following is our **target project structure**. Some files and directories have not been created yet and will be added as development progresses.

```text
cmsc-170-ai-project/
├── data/
│   ├── README.md
│   └── DATA_PLAN.md
│
├── src/
│   ├── models/
│   │   ├── linear_regression.py
│   │   ├── decision_tree.py
│   │   └── random_forest.py
│   │
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── evaluation.py
│
├── results/
│
├── docs/
│   └── index.html
│
├── README.md
├── CONTRIBUTIONS.md
└── .gitignore
```

### Directory Descriptions

| Path | Purpose |
|---|---|
| `data/` | Dataset documentation, sources, and data preparation plan |
| `src/models/` | From-scratch implementations of the three regression methods |
| `src/data_loader.py` | Loading and reading dataset records |
| `src/preprocessing.py` | Data cleaning, feature encoding, and dataset splitting |
| `src/evaluation.py` | Evaluation metrics and model comparison |
| `results/` | Experimental results, comparison tables, and visualizations |
| `docs/` | GitHub Pages project website |
| `CONTRIBUTIONS.md` | Group responsibilities and contribution log |
| `.gitignore` | Excludes local datasets and unnecessary files from Git tracking |

**Note:** Dataset files are excluded from Git tracking. The original dataset can be obtained from the source linked above. Empty directories may not appear on GitHub until files are added.

## Implementation Guidelines

All machine learning methods will be implemented from scratch by the group.

Permitted tools include:

- Python standard library
- NumPy (arrays and basic linear algebra only)
- Matplotlib (plots and visualizations)

Libraries that provide prebuilt machine learning algorithms, such as scikit-learn, TensorFlow, and PyTorch, will not be used.

All group members are responsible for understanding their implementations and contributing to the project's development.

## Development Workflow

The project follows a collaborative Git workflow:

1. The `main` branch contains the integrated project.
2. Members work on separate feature branches for their assigned tasks.
3. Changes are reviewed and integrated through pull requests.
4. All implementations follow the agreed data format and experimental setup.
5. Contributions and milestone progress are documented throughout development.

### Initial Responsibilities

| Member | Primary Responsibility |
|---|---|
| Group Leader | Repository management, data preparation, Linear Regression, and integration |
| Member 2 | Decision Tree Regression |
| Member 3 | Random Forest Regression |

Assignments may be adjusted based on group discussion and development requirements.

All members will contribute to testing, evaluation, documentation, and the final presentation.

## Project Milestones

| Milestone | Schedule | Deliverable |
|---|---|---|
| 1 — Proposal | October 19–23, 2026 | Dataset and methods approval |
| 2 — Working Program | November 2–6, 2026 | Working implementations of all methods |
| 3 — Comparison | November 9–13, 2026 | Evaluation workflow and comparison results |
| 4 — Project Page | November 16–20, 2026 | GitHub Pages project website |
| 5 — Presentation | November 23–27, 2026 | Final group presentation |

## Reproducibility

Instructions for dataset preparation, running the implementations, configuring model parameters, and reproducing experimental results will be documented as development progresses.

The Parquet-to-CSV conversion procedure is subject to confirmation with the course instructor.

## References and AI Assistance

- DPWH Infrastructure Transparency Dataset — https://data.bettergov.ph/
- CMSC 170 Final Project Guidelines

ChatGPT has been used to assist with initial dataset exploration, project planning, repository organization, and documentation.

All machine learning implementations will be written and reviewed by the group. Additional AI assistance, if any, will be documented in the final project page.
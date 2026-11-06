# Shared Data Preprocessing

This module prepares the DPWH Infrastructure Transparency Dataset
for the three regression algorithms.

## Dataset Preparation

- Valid projects: 241,155
- Training data: 192,924 projects (80%)
- Testing data: 48,231 projects (20%)
- Random seed: 67

The same training and testing split is used by all three algorithms.

## Input Features

Each project has 25 numerical features:

- 6 category features: 5 most common categories + Other
- 18 region features: one-hot encoded regions
- 1 infrastructure year feature: scaled using the training year range

Categories and regions are encoded without assigning them a
numerical order.

The category selection, region encoding, and year scaling are
based on training data only to avoid data leakage.

The target value is the project's actual budget in PHP.

## How to Use

Import the shared preprocessing function:

```python
from src.preprocessing import prepare_data

X_train, y_train, X_test, y_test = prepare_data()
```

- X_train: training input features
- y_train: actual training budgets
- X_test: testing input features
- y_test: actual testing budgets

All three algorithms should use these same datasets for a
consistent comparison.

## Dataset File

The CSV dataset should be located at:

data/dpwh_transparency_data.csv

Run the project from the repository root directory.

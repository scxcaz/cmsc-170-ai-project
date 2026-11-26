# Shared Data Preprocessing

This module prepares the DPWH Infrastructure Transparency Dataset
for the three regression algorithms.

## Dataset Preparation

- Valid projects: 241,155
- Training data: 192,924 projects (80%)
- Testing data: 48,231 projects (20%)
- Random seed: 67

The same original training and testing split is used by all three algorithms. Feature and setting selection uses separate validation splits (seeds 68, 69, and 70) drawn only from the original training partition. The final evaluation uses the reserved 48,231 projects.

## Input Features

The final comparison uses 152 numerical features per project.

| Group | Features | Encoding |
| --- | ---: | --- |
| Project category | 6 | Five most common categories plus Other |
| Region | 18 | One-hot encoding of training regions |
| Infrastructure year | 1 | Min-max scaling using the training year range |
| Project description | 10 | Binary indicators from a fixed list of construction activities and infrastructure terms |
| Detailed location | 101 | Top 100 normalized DPWH administrative labels plus Other |
| Source of funds | 16 | Top 15 normalized funding programs plus Other |

Feature vocabularies and scaling parameters are derived from training records. Location labels are administrative descriptors, including labels beyond literal province names. Embedded years are removed from funding labels because infrastructure year is already represented. Availability of funding-program identity before budget approval must be checked for any early-stage use.

The target is the project's recorded budget in PHP. The original 25-feature representation consists of the first three groups.

## How to Use

Import the shared preprocessing function:

```python
from src.preprocessing import prepare_data

X_train, y_train, X_test, y_test = prepare_data(
    include_description=True,
    include_location=True,
    include_funding=True,
)
```

- X_train: training input features
- y_train: actual training budgets
- X_test: testing input features
- y_test: actual testing budgets

All three algorithms should use these same datasets for a
consistent comparison. Calling `prepare_data()` with its defaults returns the original 25-feature representation. The final evaluation does not enable validation mode.

## Dataset File

The CSV dataset should be located at:

data/dpwh_transparency_data.csv

Run the project from the repository root directory.

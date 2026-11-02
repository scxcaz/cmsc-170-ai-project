# Data Preparation Plan

## Objective
Predict DPWH infrastructure project budgets using historical project characteristics.

## Dataset
- Records: 248,220
- Source format: Parquet
- Working format: CSV

## Proposed Features
- category
- infraYear
- location.region

## Target
- budget (PHP)

## Initial Preprocessing Plan
1. Load CSV using Python's built-in csv module.
2. Exclude records with zero or invalid budgets.
3. Extract region from the location field.
4. Convert categorical features into numerical representations.
5. Split data into training (80%) and testing (20%).
6. Apply the same prepared dataset to all three algorithms.

## Important Considerations
- Use a fixed random seed for reproducibility.
- Prevent data leakage during preprocessing.
- Start with a manageable dataset subset for development.
- Review outliers and feature selection before final evaluation.
- Document all preprocessing decisions.

## Status
Preliminary plan for group review.

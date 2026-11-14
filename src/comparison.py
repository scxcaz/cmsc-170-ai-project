import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing import prepare_data
from src import evaluation

from src.models.linear_regression import (
    train,
    predict_batch,
    budgets_to_millions,
    millions_to_pesos,
)
from src.models.decision_tree import DecisionTreeRegressor
from src.models.random_forest import RandomForestRegressor


# calculates the four shared metrics for a model's predictions
def evaluate_model(actual, predicted):
    return {
        "MAE": evaluation.mean_absolute_error(actual, predicted),
        "MSE": evaluation.mean_squared_error(actual, predicted),
        "RMSE": evaluation.root_mean_squared_error(actual, predicted),
        "R2": evaluation.r_squared(actual, predicted),
    }


# displays the evaluation results in a comparison table
def display_results(results):
    print("\nMODEL COMPARISON - TESTING DATA")

    header = (
        f"{'Model':<20} | "
        f"{'MAE (PHP)':>16} | "
        f"{'MSE (PHP^2)':>14} | "
        f"{'RMSE (PHP)':>16} | "
        f"{'R2':>7}"
    )

    print(header)
    print("-" * len(header))

    # displays the metrics for each regression model
    for name, metrics in results.items():
        print(
            f"{name:<20} | "
            f"{metrics['MAE']:>16,.2f} | "
            f"{metrics['MSE']:>14.3e} | "
            f"{metrics['RMSE']:>16,.2f} | "
            f"{metrics['R2']:>7.4f}"
        )


# saves the results as a CSV file
def save_results(results, output_dir):
    file_path = output_dir / "model_comparison.csv"

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # writes the column headings and each model's results
        writer.writerow(["Model", "MAE_PHP", "MSE_PHP2", "RMSE_PHP", "R2"])

        for name, metrics in results.items():
            writer.writerow([
                name,
                metrics["MAE"],
                metrics["MSE"],
                metrics["RMSE"],
                metrics["R2"],
            ])


# creates graphs to compare the regression models
def create_graphs(results, output_dir):
    names = list(results.keys())
    positions = np.arange(len(names))

    # converts monetary errors into millions of PHP for readability
    mae_values = [results[name]["MAE"] / 1_000_000 for name in names]
    rmse_values = [results[name]["RMSE"] / 1_000_000 for name in names]
    r2_values = [results[name]["R2"] for name in names]

    # compares MAE and RMSE in the same graph
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(positions - width / 2, mae_values, width, label="MAE")
    plt.bar(positions + width / 2, rmse_values, width, label="RMSE")

    plt.xticks(positions, names)
    plt.ylabel("Error (million PHP)")
    plt.title("Regression Model Error Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "error_comparison.png", dpi=150)
    plt.close()

    # compares how much variation each model explains
    plt.figure(figsize=(9, 6))
    plt.bar(names, r2_values)
    plt.axhline(0)
    plt.ylabel("R-squared")
    plt.title("Regression Model R-squared Comparison")
    plt.tight_layout()

    plt.savefig(output_dir / "r2_comparison.png", dpi=150)
    plt.close()


def main():
    # loads the shared dataset and converts it into NumPy arrays
    X_train, y_train, X_test, y_test = prepare_data()

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    print("Training projects:", len(y_train))
    print("Testing projects:", len(y_test))

    results = {}

    # trains linear regression using budgets in millions of PHP
    print("\nTraining Linear Regression...")
    print("Learning rate: 0.05 | Iterations: 200")

    weights, bias = train(
        X_train, budgets_to_millions(y_train),
        learning_rate=0.05, iterations=200
    )

    # converts predictions back to PHP before evaluation
    linear_predictions = millions_to_pesos(
        predict_batch(X_test, weights, bias)
    )

    results["Linear Regression"] = evaluate_model(
        y_test, linear_predictions
    )

    # trains decision tree regression using the original PHP budgets
    print("\nTraining Decision Tree...")
    print("Depth: 16 | Min split: 20 | Min leaf: 10")

    decision_tree = DecisionTreeRegressor(
        max_depth=16,
        min_samples_split=20,
        min_samples_leaf=10,
    )

    decision_tree.fit(X_train, y_train)
    tree_predictions = decision_tree.predict(X_test)

    results["Decision Tree"] = evaluate_model(
        y_test, tree_predictions
    )

    # trains random forest using the verified hyperparameters
    print("\nTraining Random Forest...")
    print("Trees: 5 | Depth: 8 | Min split: 20 | Min leaf: 10")
    print("Features per split: 5 | Random state: 42")

    random_forest = RandomForestRegressor(
        n_trees=5,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=5,
        random_state=42,
    )

    random_forest.fit(X_train, y_train)
    forest_predictions = random_forest.predict(X_test)

    results["Random Forest"] = evaluate_model(
        y_test, forest_predictions
    )

    # displays the model comparison and the median baseline
    display_results(results)

    median_budget = np.median(y_train)
    median_predictions = np.full_like(y_test, median_budget)
    baseline_mae = evaluation.mean_absolute_error(
        y_test, median_predictions
    )

    print(f"\nMedian baseline MAE: PHP {baseline_mae:,.2f}")

    # creates the folder for the comparison files
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # saves the numerical results and comparison graphs
    save_results(results, output_dir)
    create_graphs(results, output_dir)

    print("\nComparison results and graphs saved in results/")


if __name__ == "__main__":
    main()